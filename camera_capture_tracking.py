import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained MNIST model
model = tf.keras.models.load_model("mnist_cnn_model.keras")

cap = cv2.VideoCapture(0)

captured_frame = None
analyzed = False


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def find_largest_rectangle(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > max_area:
                biggest = approx
                max_area = area

    return biggest


def make_mnist_style(img):
    h, w = img.shape

    # Scale so max dimension = 20
    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Center it
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


while True:
    if not analyzed:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        frame = captured_frame.copy()

    display_frame = frame.copy()

    if analyzed:
        thresh = preprocess_image(frame)
        box = find_largest_rectangle(thresh)

        if box is not None:
            cv2.drawContours(display_frame, [box], -1, (255, 0, 0), 3)

            x, y, w, h = cv2.boundingRect(box)
            roi = thresh[y : y + h, x : x + w]

            contours, _ = cv2.findContours(
                roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            digit_boxes = []

            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for i, contour in enumerate(contours):
                if i == 0:
                    continue  # Skip the bounding rectangle contour

                if cv2.contourArea(contour) < 100:
                    continue

                dx, dy, dw, dh = cv2.boundingRect(contour)
                digit_boxes.append((dx, dy, dw, dh))

            digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

            recognized_digits = []

            for dx, dy, dw, dh in digit_boxes:
                digit_image = roi[dy : dy + dh, dx : dx + dw]

                digit_image = make_mnist_style(digit_image)
                digit_image = digit_image / 255.0
                digit_image = np.expand_dims(digit_image, axis=-1)
                digit_image = np.expand_dims(digit_image, axis=0)

                prediction = model.predict(digit_image, verbose=0)
                digit = np.argmax(prediction)
                recognized_digits.append(str(digit))

                cv2.rectangle(
                    display_frame,
                    (x + dx, y + dy),
                    (x + dx + dw, y + dy + dh),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display_frame,
                    str(digit),
                    (x + dx, y + dy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            if recognized_digits:
                number_str = "".join(recognized_digits)
                cv2.putText(
                    display_frame,
                    f"Number: {number_str}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
        else:
            cv2.putText(
                display_frame,
                "No box detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    cv2.putText(
        display_frame,
        "C: Capture  R: Reset  Q: Quit",
        (10, display_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Digit Recognition Demo", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c") and not analyzed:
        captured_frame = frame.copy()
        analyzed = True

    elif key == ord("r"):
        analyzed = False

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
