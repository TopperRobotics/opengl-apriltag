#include "GpuDetector.hpp"

#include <GL/glew.h>
#include <string>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <limits>

namespace {

std::string readShaderSource(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Failed to open shader: " + path);
    std::stringstream buf;
    buf << file.rdbuf();
    return buf.str();
}

GLuint compileShaderFromFile(const std::string& path) {
    std::string src = readShaderSource(path);
    const char* csrc = src.c_str();
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &csrc, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(static_cast<size_t>(len + 1), 0);
        glGetShaderInfoLog(s, len, nullptr, log.data());
        glDeleteShader(s);
        throw std::runtime_error("Shader compile failed (" + path + "): " + log.data());
    }
    return s;
}

GLuint linkComputeProgram(GLuint shader) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader);
    glLinkProgram(prog);
    glDeleteShader(shader);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(static_cast<size_t>(len + 1), 0);
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        glDeleteProgram(prog);
        throw std::runtime_error(std::string("Shader link failed: ") + log.data());
    }
    return prog;
}

GLuint loadComputeProgram(const std::string& shaderDir, const char* filename) {
    return linkComputeProgram(compileShaderFromFile(shaderDir + "/" + filename));
}

void dispatch2D(int w, int h, int localX = 16, int localY = 16) {
    int gx = (w + localX - 1) / localX;
    int gy = (h + localY - 1) / localY;
    glDispatchCompute(static_cast<GLuint>(gx), static_cast<GLuint>(gy), 1);
}

} // anonymous namespace

struct GpuDetector::Impl {
    Config cfg;
    std::string shaderDir;

    GLuint thresholdProg = 0;
    GLuint cclInitProg = 0;
    GLuint cclMergeProg = 0;
    GLuint cclFinalProg = 0;

    GLuint inputTex = 0;
    GLuint binaryImg = 0;
    GLuint labelImgA = 0;
    GLuint labelImgB = 0;

    GLuint ufParentBuf = 0;

    int width = 0;
    int height = 0;
    int activeComponents = 0;

    std::vector<ComponentInfo> compHost;
    std::vector<uint32_t> boundaryHost;

    void destroyGL() {
        GLuint textures[] = {inputTex, binaryImg, labelImgA, labelImgB};
        glDeleteTextures(4, textures);
        inputTex = binaryImg = labelImgA = labelImgB = 0;

        if (ufParentBuf) glDeleteBuffers(1, &ufParentBuf);
        ufParentBuf = 0;

        GLuint progs[] = {thresholdProg, cclInitProg, cclMergeProg, cclFinalProg};
        glDeleteProgram(thresholdProg);
        glDeleteProgram(cclInitProg);
        glDeleteProgram(cclMergeProg);
        glDeleteProgram(cclFinalProg);
        thresholdProg = cclInitProg = cclMergeProg = cclFinalProg = 0;

        width = height = 0;
        activeComponents = 0;
        compHost.clear();
        boundaryHost.clear();
    }

    void allocateResources(int w, int h) {
        destroyGL();

        width = w;
        height = h;

        glGenTextures(1, &inputTex);
        glBindTexture(GL_TEXTURE_2D, inputTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

        glGenTextures(1, &binaryImg);
        glBindTexture(GL_TEXTURE_2D, binaryImg);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, w, h, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);

        glGenTextures(1, &labelImgA);
        glBindTexture(GL_TEXTURE_2D, labelImgA);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, w, h, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

        glGenTextures(1, &labelImgB);
        glBindTexture(GL_TEXTURE_2D, labelImgB);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, w, h, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

        const size_t ufCount = static_cast<size_t>(w) * static_cast<size_t>(h) + 1;
        glGenBuffers(1, &ufParentBuf);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ufParentBuf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, ufCount * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        boundaryHost.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
        compHost.clear();
    }

    void extractComponentsCPU(const cv::Mat& gray, const std::vector<uint32_t>& labels) {
        struct Accum {
            int count = 0;
            int minX = std::numeric_limits<int>::max();
            int minY = std::numeric_limits<int>::max();
            int maxX = std::numeric_limits<int>::min();
            int maxY = std::numeric_limits<int>::min();
            double sumX = 0.0;
            double sumY = 0.0;
            std::vector<cv::Point2f> border;
        };

        std::unordered_map<uint32_t, Accum> accum;
        const int w = width;
        const int h = height;

        auto isBlack = [&](int x, int y) {
            return gray.at<uint8_t>(y, x) == 0;  // or < 128
        };

        auto isBorder = [&](int x, int y, uint32_t label) {
            if (!isBlack(x, y)) return false;
            const int dx[4] = {-1, 1, 0, 0};
            const int dy[4] = {0, 0, -1, 1};
            for (int i = 0; i < 4; ++i) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) return true;
                if (labels[static_cast<size_t>(ny * w + nx)] != label) return true;
            }
            return false;
        };

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                uint32_t label = labels[static_cast<size_t>(y * w + x)];
                if (label == 0) continue;
                auto& a = accum[label];
                a.count++;
                a.minX = std::min(a.minX, x);
                a.minY = std::min(a.minY, y);
                a.maxX = std::max(a.maxX, x);
                a.maxY = std::max(a.maxY, y);
                a.sumX += x;
                a.sumY += y;
                if (isBorder(x, y, label)) {
                    a.border.emplace_back(static_cast<float>(x), static_cast<float>(y));
                }
            }
        }

        compHost.clear();
        boundaryHost.clear();
        activeComponents = 0;

        for (auto& [label, a] : accum) {
            if (a.count < cfg.minTagArea) continue;
            int bw = a.maxX - a.minX + 1;
            int bh = a.maxY - a.minY + 1;
            float aspect = static_cast<float>(bw) / static_cast<float>(std::max(1, bh));
            if (aspect < 0.5f || aspect > 2.0f) continue;

            ComponentInfo info{};
            info.count = static_cast<uint32_t>(a.border.size());
            info.offset = static_cast<uint32_t>(boundaryHost.size());
            info.bboxMinX = a.minX;
            info.bboxMinY = a.minY;
            info.bboxMaxX = a.maxX;
            info.bboxMaxY = a.maxY;
            info.centroidX = static_cast<float>(a.sumX / std::max(1, a.count));
            info.centroidY = static_cast<float>(a.sumY / std::max(1, a.count));

            for (const auto& pt : a.border) {
                uint32_t packed = static_cast<uint32_t>(pt.x) | (static_cast<uint32_t>(pt.y) << 16);
                boundaryHost.push_back(packed);
            }

            compHost.push_back(info);
            activeComponents++;
            (void)label;
        }
    }
};

GpuDetector::GpuDetector(const Config& cfg)
    : config_(cfg), impl_(std::make_unique<Impl>()) {
    impl_->cfg = cfg;
}

GpuDetector::~GpuDetector() {
    if (impl_) impl_->destroyGL();
}

bool GpuDetector::compileShaders(const std::string& shaderDir) {
    impl_->shaderDir = shaderDir;
    impl_->thresholdProg = loadComputeProgram(shaderDir, "adaptive_threshold.comp");
    impl_->cclInitProg = loadComputeProgram(shaderDir, "ccl_init.comp");
    impl_->cclMergeProg = loadComputeProgram(shaderDir, "ccl_merge.comp");
    impl_->cclFinalProg = loadComputeProgram(shaderDir, "ccl_final.comp");
    return true;
}

bool GpuDetector::resize(int width, int height) {
    if (width <= 0 || height <= 0) return false;
    config_.width = width;
    config_.height = height;
    impl_->cfg.width = width;
    impl_->cfg.height = height;
    impl_->allocateResources(width, height);
    return true;
}

int GpuDetector::dispatch(const cv::Mat& grayImage) {
    if (grayImage.empty() || grayImage.type() != CV_8UC1) return 0;
    if (impl_->thresholdProg == 0) return 0;

    const int w = grayImage.cols;
    const int h = grayImage.rows;
    if (impl_->width != w || impl_->height != h) {
        resize(w, h);
    }

    auto* impl = impl_.get();

    glBindTexture(GL_TEXTURE_2D, impl->inputTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, grayImage.data);

    const int imageSize[2] = {w, h};

    // Pass 1: adaptive threshold
    glUseProgram(impl->thresholdProg);
    std::cout << "here here here here!!!!" << std::endl;
    glBindImageTexture(0, impl->inputTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
    glBindImageTexture(1, impl->binaryImg, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8UI);
    glUniform1f(glGetUniformLocation(impl->thresholdProg, "u_thresholdConst"), config_.thresholdConst);
    glUniform1i(glGetUniformLocation(impl->thresholdProg, "u_windowSize"), config_.windowSize);
    glUniform2iv(glGetUniformLocation(impl->thresholdProg, "u_imageSize"), 2, imageSize);
    dispatch2D(w, h);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    // write pass 1 to disk for debugging
    std::vector<uint8_t> binaryData(static_cast<size_t>(w * h));
    glBindTexture(GL_TEXTURE_2D, impl->binaryImg);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, binaryData.data());
    cv::Mat binaryMat(h, w, CV_8UC1, binaryData.data());
    cv::imwrite("binary_output.png", binaryMat);

    // Reset union-find buffer
    const size_t ufCount = static_cast<size_t>(w) * static_cast<size_t>(h) + 1;
    std::vector<uint32_t> ufInit(ufCount, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, impl->ufParentBuf);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ufInit.size() * sizeof(uint32_t), ufInit.data());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, impl->ufParentBuf);

    // Pass 2a: CCL init
    glUseProgram(impl->cclInitProg);
    glBindImageTexture(0, impl->binaryImg, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8UI);
    glBindImageTexture(1, impl->labelImgA, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glUniform2iv(glGetUniformLocation(impl->cclInitProg, "u_imageSize"), 2, imageSize);
    dispatch2D(w, h);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    // Pass 2b: CCL merge (multiple iterations)
    glUseProgram(impl->cclMergeProg);
    glBindImageTexture(0, impl->labelImgA, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glUniform2iv(glGetUniformLocation(impl->cclMergeProg, "u_imageSize"), 2, imageSize);
    for (int i = 0; i < 5; ++i) {
        dispatch2D(w, h);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // Pass 2c: CCL final resolve
    glUseProgram(impl->cclFinalProg);
    glBindImageTexture(0, impl->labelImgA, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI);
    glBindImageTexture(1, impl->labelImgB, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
    glUniform2iv(glGetUniformLocation(impl->cclFinalProg, "u_imageSize"), 2, imageSize);
    dispatch2D(w, h);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    // Read back label image for CPU component extraction
    std::vector<uint32_t> labels(static_cast<size_t>(w * h));
    glBindTexture(GL_TEXTURE_2D, impl->labelImgB);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, labels.data());
    std::cout << "Label image read back, first 10 labels: ";
    for (int i = 0; i < std::min(10, static_cast<int>(labels.size())); ++i) {
        std::cout << labels[i] << " ";
    }
    std::cout << std::endl;

    // Build binary-ish image from threshold output for border detection
    cv::Mat binary(h, w, CV_8UC1);
    glBindTexture(GL_TEXTURE_2D, impl->binaryImg);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, binary.data);
    impl->extractComponentsCPU(binary, labels);
    impl->activeComponents = static_cast<int>(impl->compHost.size());
    return impl->activeComponents;
}

std::vector<GpuDetector::ComponentInfo> GpuDetector::getComponentInfo() const {
    return impl_->compHost;
}

std::vector<uint32_t> GpuDetector::getBorderPoints() const {
    return impl_->boundaryHost;
}
