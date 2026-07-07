To do:
 
1. NetworkTables  
2. Field Relative Localization

Build it with mkdir -p build && cd build && cmake .. && cmake --build . -j$(nproc) \
Run it with ./apriltag --camera 0 --db ../data/config.db --shaders ../shaders --port 8008 --webui ../webui --camera-stream-port 8009 --snapshot ../snapshots --calibration ../calibration
