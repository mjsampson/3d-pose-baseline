FROM tensorflow/tensorflow:1.12.0-gpu

RUN apt-get update -y
RUN apt-get upgrade -y
RUN pip install h5py
RUN pip install matplotlib
RUN apt-get install python-tk -y


#RUN python3 src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 1 --sample --load 24371

#RUN python3 /code/test.py
#Need to install dependencies here