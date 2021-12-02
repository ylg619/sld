FROM python:3.8.12-buster

COPY requirements.txt /requirements.txt
COPY streamlit_app.py /streamlit_app.py
COPY handdetector.py /handdetector.py
COPY models /models
COPY images /images

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD streamlit run streamlit_app.py  --server.port $PORT --server.address 0.0.0.0