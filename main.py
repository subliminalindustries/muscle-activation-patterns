import sys
import cv2
from scipy.fft import irfft, rfft
from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def frame_by_frame(video_file):
    vid = cv2.VideoCapture(video_file)
    if vid.isOpened() is False:
        raise RuntimeError('Could not read video file')

    cv2.namedWindow('frame')
    cv2.namedWindow('selection')

    print('loading video..')

    frames = []
    detector_vals = {}

    i = 0
    while vid.isOpened():
        if i % 10000 == 0:
            print(f'frame {i}')

        ret, frame = vid.read()
        if ret is False:
            break

        frames.append(frame)

        i += 1

    i = 0
    origin = None
    size = None
    size_scalar = 100
    vert_max = 0.
    show_eyes = False
    record_detector = False
    color_white = (255, 255, 255)
    spect_mode = 0
    spect_modes = ('psd', 'magnitude', 'angle', 'phase')
    while True:
        print(f'frame {i}')
        work = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
        frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2RGBA)
        if origin is None:
            origin = [frame.shape[1] // 2, frame.shape[0] // 2]
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

        cv2.rectangle(frame, origin, np.subtract(size, 1), color_white)
        detector_origin = (origin[0] + (size_scalar // 2) - 1, origin[1])
        detector_size = (size[0] - (size_scalar // 2) - 1, size[1] - 1)
        cv2.line(frame, detector_origin, detector_size, color_white, 1)
        cv2.putText(frame, f'frame: {i}', (5, 25), 2, .5, color_white)
        cv2.putText(frame, f'selection: x1={origin[0]},x2={size[0]},y1={origin[1]},y2={size[1]}', (5, 50), 2, .5,
                    color_white)
        cv2.putText(frame, f'detect eyes: {repr(show_eyes)}', (5, 75), 2, .5, color_white)
        cv2.putText(frame, f'record detector: {repr(record_detector)}', (5, 100), 2, .5, color_white)

        selection = work[origin[1]:size[1], origin[0]:size[0]]
        detector = np.array(selection[:, selection.shape[0] // 2, 2], dtype=np.float64)
        detector -= np.min(detector)
        detector /= np.ptp(detector)
        if record_detector:
            detector_vals[i] = np.mean(detector)
        cv2.putText(frame, f'detector mean: {detector_vals.get(i)}', (5, 125), 2, .5, color_white)

        if show_eyes:
            try:
                sel_gray = cv2.cvtColor(selection, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(sel_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(selection, (ex, ey), (ex + ew, ey + eh), (0, 225, 255), 2)
            except cv2.error as e:
                print(e)

        cv2.imshow('frame', frame)
        cv2.imshow('selection', selection)

        key = cv2.waitKey()
        print(key)
        if key == 27:
            break

        if key == 69:
            show_eyes = show_eyes is False
            # toggle eye detection
            continue

        if key == 97:
            origin[0] = max(0, origin[0] - 1)
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box left
            continue

        if key == 100:
            origin[0] = min(frame.shape[1], origin[0] + 1)
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box right
            continue

        if key == 115:
            origin[1] = min(frame.shape[0], origin[1] + 1)
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box down
            continue

        if key == 119:
            origin[1] = max(0, origin[1] - 1)
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box up
            continue

        if key == 65:
            origin[0] = max(0, origin[0] - 10)
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box left shift
            continue

        if key == 68:
            origin[0] = min(frame.shape[1], origin[0] + 10)
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box right
            continue

        if key == 83:
            origin[1] = min(frame.shape[0], origin[1] + 10)
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box down
            continue

        if key == 87:
            origin[1] = max(0, origin[1] - 10)
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box up
            continue

        if key == 113:
            size_scalar = max(size_scalar // 2, 40)
            print(f'box size: {size_scalar}')
            size = [origin[0] + size_scalar, origin[1] + size_scalar]
            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box decrease size
            continue

        if key == 101:
            size_scalar = min(size_scalar * 2, min(frame.shape[:2]))
            print(f'box size: {size_scalar}')
            size = [origin[0] + size_scalar, origin[1] + size_scalar]

            size[0] = min(size[0], frame.shape[1])
            size[1] = min(size[1], frame.shape[0])
            # box increase size
            continue

        if key == 2:
            i = max(0, i - 1)
            continue

        if key == 3:
            i = min(len(frames) - 1, i + 1)
            continue

        if key == 91:
            i = max(0, i - 100)
            continue

        if key == 93:
            i = min(len(frames) - 1, i + 100)
            continue

        if key == 114:
            record_detector = record_detector is False
            continue

        if key == 109:
            spect_mode += 1
            if spect_mode > 3:
                spect_mode = 0

            print(f'spectrum mode: {spect_modes[spect_mode]}')

        if key == 102:
            if not len(detector_vals) > 100:
                print('less than 100 detector values')
                continue

            frame_rate = 24.10485
            record_detector = False
            n_frames = len(frames)
            times = np.linspace(0., (1/frame_rate) * n_frames, n_frames)
            time_series = {}
            vals = []
            for _, v in detector_vals.items():
                vals.append(v)

            vals = np.array(vals, dtype=np.float64)
            vals -= np.min(vals)
            vals /= np.ptp(vals)
            for k, v in enumerate(vals):
                time_series[times[k]] = v

            f, t, Sxx = spectrogram(vals, frame_rate, 'hann', mode=spect_modes[spect_mode])
            plt.pcolormesh(t, f, Sxx, shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()

            continue

        i += 1


if __name__ == '__main__':
    frame_by_frame(sys.argv[1])
