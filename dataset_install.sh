#!/bin/bash
set -euf -o pipefail

cd data
gdown https://drive.google.com/uc?id=1rd202o1utgy_QVH1wMLzg17hCyrFDtlh
gdown https://drive.google.com/uc?id=1r2ePb234oN82gnoOc2UTX58X0KaBGcXM
gdown https://drive.google.com/uc?id=1Dwk9yp29V7ccVoSjPmJjOxIwW7EXc5cv
gdown https://drive.google.com/uc?id=1qhE8bcCZ4LlwVoFkq9h8BnE36ITzb6_y
gdown https://drive.google.com/uc?id=1cTlH70qwGA7GlDG4JUAtSgqVOCqApAn_
gdown https://drive.google.com/uc?id=1Lrcq-CdWIuaU7Zk7gSr-XghilSNsZV0H
gdown https://drive.google.com/uc?id=1ftuNJSZXt98-8VR34bCl87UdWYx46NkM
gdown https://drive.google.com/uc?id=1eVZR6k9Tb-9hhQDaILZyQUYI80kp5pLz
gdown https://drive.google.com/uc?id=1uApezJ_Jtcw4tKFvAQ05JNHlaEs6b0L8

unzip -d vibe_dilated vibe_dilated.zip
unzip -d vibe_undilated vibe_undilated.zip