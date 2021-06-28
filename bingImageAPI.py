from bing_image_downloader import downloader

downloader.download("karim benzema", limit=20,  output_dir='dataset', adult_filter_off=True, force_replace=False)
downloader.download("cristiano ronaldo", limit=20,  output_dir='dataset', adult_filter_off=True, force_replace=False)
downloader.download("lionel messi", limit=20,  output_dir='dataset', adult_filter_off=True, force_replace=False)
downloader.download("eden hazard", limit=20,  output_dir='dataset', adult_filter_off=True, force_replace=False)