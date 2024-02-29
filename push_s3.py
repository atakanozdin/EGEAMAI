import os
import boto3
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_folder, bucket_name, s3_folder_name):
    # AWS Access ve Secret Key bilgilerinizi buraya ekleyin
    access_key = 'AKIAQ3EGVYZCPYZT54ZM'
    secret_key = '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'

    # Boto3 S3 client oluşturun
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    try:
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_file_name = os.path.relpath(local_file_path, local_folder)
                s3_file_path = os.path.join(s3_folder_name, s3_file_name)
                s3.upload_file(local_file_path, bucket_name, s3_file_path)
                print(f"{local_file_path} dosyası başarıyla yüklendi.")
        return True
    except FileNotFoundError:
        print("Belirtilen dosya bulunamadı.")
        return False
    except NoCredentialsError:
        print("AWS kimlik bilgileri geçerli değil.")
        return False
    
# Kullanım örneği
local_folder_path = 'D:/EgeAMAI/Documents'  # Yerel klasörün yolu
bucket_name = 'egeamaipdf'         # S3 bucket adı
s3_folder_name = 'PDF_Documents'        # S3'de oluşturulacak klasörün adı

upload_to_s3(local_folder_path, bucket_name, s3_folder_name)