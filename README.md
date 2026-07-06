To do:

1. CHArUco Camera Calibration  
2. NetworkTables  
3. Field Relative Localization

Build it with mkdir -p build && cd build && cmake .. && cmake --build . -j$(nproc)
Run it with ./apriltag --camera 0 --db ../data/config.db --shaders ../shaders --port 8008 --webui ../webui --camera-stream-port 8009
