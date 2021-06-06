from minio import Minio
from minio.error import ResponseError

from io import BytesIO

host="localhost:9000"
access_key="minioadmin"
secret_key="minioadmin"

minioClient=Minio(host,access_key=access_key,secret_key=secret_key,secure=False)

video="My minio content"
bucket="video"
content=BytesIO(bytes(video,'utf-8'))
key='video.mp4'
size=content.getbuffer().nbytes

try:
    minioClient.put_object(bucket,key,content,size)
    print("uploaded")
except ResponseError as err:
    print("error",err)    
