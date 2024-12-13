import boto3
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import List
from urllib.parse import urlparse
from botocore.exceptions import ClientError


def parse_s3_url(url: str) -> tuple[str, str]:
    """
    Parse an S3 URL into bucket name and key.
    Now with additional debug printing.
    """
    parsed = urlparse(url)
    
    if parsed.scheme == 's3':
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
    else:  # http or https
        if '.s3.amazonaws.com' in parsed.netloc:
            bucket = parsed.netloc.split('.s3.amazonaws.com')[0]
        else:
            bucket = parsed.netloc.split('.')[0]  # Take first part before any dots
        
        key = parsed.path.lstrip('/')
    
    print(f"Debug - Parsed URL: {url}")
    print(f"Debug - Bucket: {bucket}")
    print(f"Debug - Key: {key}")
    
    return bucket, key

def download_images_from_s3(urls: List[str], local_dir: str, max_workers: int = 5, 
                          aws_access_key_id: str = None,
                          aws_secret_access_key: str = None,
                          region_name: str = None) -> List[str]:
    """
    Download images from S3 URLs with explicit credentials support.
    """
    # Initialize S3 client with optional credentials
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name or 'us-east-1'  # Default to us-east-1 if not specified
    )
    s3_client = session.client('s3')
    
    os.makedirs(local_dir, exist_ok=True)
    downloaded_files = []
    
    def download_single_image(url: str) -> str:
        try:
            # Parse bucket and key from URL
            bucket, key = parse_s3_url(url)
            
            # Create local file path
            local_file = os.path.join(local_dir, os.path.basename(key))
            
            # Try to check if object exists first
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', 'Unknown error')
                print(f"Error checking object existence - Code: {error_code}, Message: {error_message}")
                print(f"Full error response: {e.response}")
                return None
            
            # If we get here, object exists, try to download
            s3_client.download_file(bucket, key, local_file)
            print(f"Successfully downloaded: {url} -> {local_file}")
            return local_file
            
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Error response: {e.response}")
            return None
    
    # Download images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_single_image, urls))
    
    # Filter out failed downloads
    downloaded_files = [f for f in results if f is not None]
    
    return downloaded_files

# Example usage
if __name__ == "__main__":
    # Example URLs in different formats
    df = pd.read_csv('/userdata/karolina.dabkowska/mask2features/caliber_20_claims.csv')
    df_clean = df.drop_duplicates(subset='claim_id')
    df_clean['image_links'] = df_clean['image_links'].apply(lambda x: x.split(','))
    df_clean = df_clean.explode('image_links')

    image_urls = df_clean['image_links'].unique().tolist()
    image_urls = [path[:-4] for path in image_urls]
    
    # Local directory to save images
    download_directory = "/userdata/karolina.dabkowska/mask2features/data/pes_sample/"
    
    # Download the images
    downloaded_files = download_images_from_s3(
        urls=image_urls,
        local_dir=download_directory,
        aws_access_key_id='ASIA4WKXWDUXN5SQVS4N',  # Optional: Provide if not using AWS CLI/IAM role
        aws_secret_access_key='W8QhUJYVHzPg3lm26STolzOq7u5z4AfLQjNFUpp4',  # Optional: Provide if not using AWS CLI/IAM role
        region_name='eu-west-2'  # Optional: Specify your bucket's region
    )
    
    print(f"\nDownloaded {len(downloaded_files)} images successfully")