import base64
import cv2
import eventlet
import numpy as np
import socketio
import time
from io import BytesIO
from PIL import Image

sio = socketio.Server()
app = socketio.WSGIApp(sio)

# ==============================================================================
# CẤU HÌNH THAM SỐ THUẬT TOÁN & TỐI ƯU HÓA
# ==============================================================================
PORT = 11000            # Cổng kết nối Simulator Game (Đổi theo màn hình Game)
CAR_CENTER_X = 320      # Tâm xe trên khung hình chuẩn 640x360
KP = 0.0035             # Hệ số đánh lái (Lớn quá xe lắc, nhỏ quá không kịp cua)
NORMAL_THROTTLE = 0.25  # Tốc độ chân ga mặc định khi chạy thẳng (0.1 - 0.5)
SLOW_THROTTLE = 0.12    # Tốc độ chân ga khi qua ngã rẽ / cua gắt
LANE_WIDTH_EST = 150    # Ước lượng khoảng cách từ vạch đến tâm làn khi mất 1 bên vạch

# Biến toàn cục cho Trạng thái điều khiển (State Machine)
state = "NORMAL"
slowdown_start_time = 0
# ==============================================================================

@sio.on('telemetry')
def telemetry(sid, data):
    global state, slowdown_start_time

    if data:
        # 1. Nhận ảnh từ Simulator & Chuyển sang dạng BGR
        img_string = data["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        frame = np.asarray(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2. Resize khung hình về chuẩn 640x360 để cố định tọa độ ROI hình thang
        frame = cv2.resize(frame, (640, 360))

        # 3. Tiền xử lý: Làm mịn + Lọc màu HSV chuẩn xác
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Ngưỡng lọc màu mặt đường/vạch đường sáng
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 50, 255])
        binary = cv2.inRange(hsv, lower_white, upper_white)

        # Tẩy rác nhỏ và lấp lỗ hổng vạch đường (Morphology Close)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 4. Cắt vùng quan tâm (ROI) Hình Thang (Bỏ cảnh vật thừa để tăng tốc độ)
        mask = np.zeros_like(binary)
        polygon = np.array([[
            (0, 360),
            (200, 210),
            (440, 210),
            (640, 360)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        roi = cv2.bitwise_and(binary, mask)

        # 5. Phân tích Contour & Phân loại Vạch đường Trái / Phải / Ngã rẽ
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        left_lanes = []
        right_lanes = []
        intersection_detected = False

        for c in contours:
            area = cv2.contourArea(c)
            if area < 200:  # Bỏ qua các đốm nhiễu quá nhỏ
                continue

            x, y, w, h = cv2.boundingRect(c)

            # Phát hiện ngã rẽ (Vạch ngang rộng nằm ở nửa dưới ROI)
            if w > h * 1.5 and area > 500 and y > 240:
                intersection_detected = True

            # Lọc các vạch kẻ đường dọc
            if h >= w:
                center_x = x + w // 2
                if center_x < CAR_CENTER_X:
                    left_lanes.append(center_x)
                else:
                    right_lanes.append(center_x)

        # 6. Tính toán điểm mục tiêu (Target Point) & Xử lý khi mất 1 bên vạch
        target_x = CAR_CENTER_X
        has_left = len(left_lanes) > 0
        has_right = len(right_lanes) > 0

        best_left = max(left_lanes) if has_left else None
        best_right = min(right_lanes) if has_right else None

        if has_left and has_right:
            # Có đủ 2 làn -> Tâm đường là trung điểm 2 vạch
            target_x = (best_left + best_right) // 2
        elif has_left and not has_right:
            # Mất làn phải -> Giữ khoảng cách an toàn lệch sang phải
            target_x = best_left + LANE_WIDTH_EST
        elif has_right and not has_left:
            # Mất làn trái -> Giữ khoảng cách an toàn lệch sang trái
            target_x = best_right - LANE_WIDTH_EST
        else:
            # Dự phòng: Nếu mất cả 2 làn, dùng Moment khối lượng toàn vùng ROI
            M = cv2.moments(roi)
            if M["m00"] > 0:
                target_x = int(M["m10"] / M["m00"])

        # 7. Quản lý Trạng thái & Chân ga (State Machine)
        current_time = time.time()

        # Phát hiện ngã rẽ -> Giảm tốc trong 1 giây
        if intersection_detected and state == "NORMAL":
            state = "SLOWDOWN"
            slowdown_start_time = current_time

        if state == "SLOWDOWN":
            throttle = SLOW_THROTTLE
            if current_time - slowdown_start_time >= 1.0:
                state = "NORMAL"
        else:
            throttle = NORMAL_THROTTLE

        # 8. Tính toán góc đánh lái (P-Controller)
        error_x = target_x - CAR_CENTER_X
        steering_angle = float(error_x * KP)
        steering_angle = max(-1.0, min(1.0, steering_angle))  # Giới hạn [-1, 1]

        # Tự động hãm ga khi ôm cua gắt
        if abs(steering_angle) > 0.35:
            throttle = min(throttle, SLOW_THROTTLE)

        # 9. Gửi lệnh điều khiển về Game
        send_control(steering_angle, throttle)
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("---> ĐÃ KẾT NỐI THÀNH CÔNG VỚI GAME! XE ĐANG TỰ LÁI...")
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )

if __name__ == '__main__':
    print(f"Lắng nghe kết nối từ Simulator tại cổng PORT: {PORT}...")
    try:
        eventlet.wsgi.server(eventlet.listen(('', PORT)), app)
    except Exception as e:
        print(f"Lỗi khởi động Server: {e}")
