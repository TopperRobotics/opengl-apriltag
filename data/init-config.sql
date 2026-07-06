CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO config VALUES ('camera_fx', '800');
INSERT OR IGNORE INTO config VALUES ('camera_fy', '800');
INSERT OR IGNORE INTO config VALUES ('camera_cx', '640');
INSERT OR IGNORE INTO config VALUES ('camera_cy', '360');
INSERT OR IGNORE INTO config VALUES ('dist_coeffs', '[0.1,-0.2,0,0,0]');
INSERT OR IGNORE INTO config VALUES ('tag_size_m', '0.16');
INSERT OR IGNORE INTO config VALUES ('adaptive_threshold_win', '31');
INSERT OR IGNORE INTO config VALUES ('adaptive_threshold_const', '7');
INSERT OR IGNORE INTO config VALUES ('min_tag_area', '100');
INSERT OR IGNORE INTO config VALUES ('max_tag_area', '10000');
INSERT OR IGNORE INTO config VALUES ('decimate_factor', '2');
INSERT OR IGNORE INTO config VALUES ('camera_exposure', '-6');
INSERT OR IGNORE INTO config VALUES ('camera_brightness', '0.5');
INSERT OR IGNORE INTO config VALUES ('camera_contrast', '0.5');
