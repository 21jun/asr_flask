# Automatic Speech Recognition 
## FLASK - NGINX
## Wav2Vec2 (HuggingFace)


### Usage

* first, place wav2vec2 model file under `flask/checkpoint`

* seconde run command at project root dir (asr_flask)

```sh
cd flask
docker build -t 21jun/asr_flask .
```

```sh
cd nginx
docker build -t 21jun/asr_flask_nginx .
```

```sh
docker-compose up
```