docker build --rm -f Dockerfile -t repast4py:latest .
docker tag p2m4py:latest dsheeler/repast4py:latest
docker push dsheeler/repast4py:latest
