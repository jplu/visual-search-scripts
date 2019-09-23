import gzip
import math
import os
import sys
import multiprocessing
import urllib.request
import requests
import glob
import imghdr
from tqdm import tqdm
import shutil

data_path = 'metadata.json'
compressed_data_path = data_path + '.gz'
images_path = 'images'
NUM_CPU = multiprocessing.cpu_count()
metadata_url = "<amazon_archive_address>"

if not os.path.isfile(compressed_data_path):
    file_size = int(requests.head(metadata_url).headers["Content-Length"])
    header = {"Range": "bytes=%s-%s" % (0, file_size)}
    pbar_1 = tqdm(total=file_size, initial=0, unit='B', unit_scale=True, desc=metadata_url.split('/')[-1])
    r = requests.get(metadata_url, headers=header, stream=True)

    print("Start downloading metadata.json.gz")

    with open("metadata.json.gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar_1.update(1024)
    
    pbar_1.close()
    
    print("Download done")

if os.path.isfile(compressed_data_path) and not os.path.isfile(data_path):
    print("Start uncompress metadata.json.gz")

    with gzip.open('metadata.json.gz', 'r') as f_in, open('metadata.json', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    
    print("Uncompress done")

if not os.path.isdir(images_path):
    os.makedirs(images_path)

def process(line):
    data = eval(line)
    
    if 'imUrl' in data and data['imUrl'] is not None and 'categories' in data and data['imUrl'].split('.')[-1] == 'jpg' and data['asin'].isdigit():
        url = data['imUrl']

        try:
            path = os.path.join(images_path, str(int(data['asin'])) + '.jpg')

            if not os.path.isfile(path):
                urllib.request.urlretrieve(url, path)
                
                if imghdr.what(path) != 'jpeg':
                    print('Removed {} it is a {}'.format(path, imghdr.what(path)))
                    sys.stdout.flush()
                    os.remove(path)
            else:
                print(path + " already exists")
                sys.stdout.flush()
        except Exception as e:
            print("Error downloading {}".format(url))
            sys.stdout.flush()

def process_wrapper(chunk_start, chunk_size):
    with open(data_path) as f:
        f.seek(chunk_start)
        lines = f.read(chunk_size).splitlines()

        for line in lines:
            process(line)

def chunkify(size=1024*1024):
    file_end = os.path.getsize(data_path)

    with open(data_path, 'rb+') as f:
        chunk_end = f.tell()

        while True:
            chunk_start = chunk_end

            f.seek(size,1)
            f.readline()
            
            chunk_end = f.tell()
            
            yield chunk_start, chunk_end - chunk_start
            
            if chunk_end > file_end:
                break


def image_upload(blobname, filename):
    blob = bucket.blob(blobname)
    blob.upload_from_filename(filename)


pool = multiprocessing.Pool(processes=NUM_CPU)

for chunk_start, chunk_size in chunkify():
    pool.apply_async(process_wrapper, args=(chunk_start, chunk_size))

pool.close()
pool.join()

