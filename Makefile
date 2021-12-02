# Regions
MULTI_REGION=eu.gcr.io
REGION=europe-west1
PROJECT_ID=spartan-tesla-328010

DOCKER_IMAGE_NAME=sign-language
BUILD_NAME = ${MULTI_REGION}/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

dok_build_app:
	docker build -t ${BUILD_NAME} . -f ./Dockerfile

dok_run_app:
	docker run -e PORT=8080 -p 8000:8080 ${BUILD_NAME}

dok_push_gcp:
	docker push ${BUILD_NAME}

gcp_run_deploy:
	gcloud run deploy --image ${BUILD_NAME} \
                	  --platform managed \
                	  --region ${REGION}