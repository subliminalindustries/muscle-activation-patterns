# muscle-activation-patterns
Detects dominant local movement frequencies in video for detecting extraneous influence on muscle activation.

## Idea

You can detect muscle movement dominant frequencies. This is useful when certain muscles are being rhythmically triggered using neurological warfare.

You can detect which RF-frequencies are used by comparing the output of this program with the readings of a magnetometer. 

## Operation

Keyboard:

```
a           move box left
d           move box right
w           move box up
s           move box down
q           decrease box size
e           increase box size
r           toggle record mode
SHIFT+e     toggle eye tracking mode
m           change spectrogram mode
f           show spectrogram
ESC         exit application
```

Usage:

```
python ./main.py /path/to/video/file.mp4
```
