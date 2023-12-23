import fiftyone.zoo as foz
import requests
import os
from zipfile import ZipFile

DavisDatasetUrls = {
    "train480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
    "test480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip",
    # "trainFull": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip",
    # "testFull": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-Full-Resolution.zip"
}


def downloadCocoDataset(uSplit, uLabelTypes, uClasses, datasetDir="../datasets/coco2017/"):
    # To download the COCO dataset for only the "person" and "car" classes
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=uSplit,  # "train"
        label_types=uLabelTypes,  # ["detections", "segmentations"],
        classes=uClasses,
        dataset_dir=datasetDir  # ["person", "car"],
        # max_samples=50,
    )

    return dataset


def downloadDavisDataset(url, path):
    if not os.path.exists(path):
        os.makedirs(path)

    zipPath = path + '/' + url.split('/')[-1]

    if not os.path.exists(zipPath):
        r = requests.get(url, stream=True)
        with open(zipPath, mode="wb") as file:
            for chunk in r.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

    with ZipFile(zipPath) as zf:
        zf.extractall(path)


if __name__ == '__main__':
    downloadDavisDataset(DavisDatasetUrls["train480p"],
                         os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/Davis/train480p")))
    downloadDavisDataset(DavisDatasetUrls["test480p"],
                         os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/Davis/test480p")))

    # COCO dataset
    #dataset = downloadCocoDataset("train", ["segmentations"], ["person"])
    #session = fo.launch_app(dataset)
