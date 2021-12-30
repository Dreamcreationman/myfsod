import os
import xml.etree.ElementTree as ET
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager


def register_meta_voc(name, metadata, dirname, split, year, keepclasses, split_id):
    if keepclasses.startswith("all"):
        thing_classes = metadata["thing_classes"][split_id]
    if keepclasses.startswith("base"):
        thing_classes = metadata["base_classes"][split_id]
    if keepclasses.startswith("novel"):
        thing_classes = metadata["novel_classes"][split_id]

    DatasetCatalog.register(name, lambda: load_filtered_voc_instance(name, dirname, split, thing_classes))
    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=dirname,
        year=year,
        split=split,
        base_classes=metadata["base_classes"][split_id],
        novel_classes=metadata["novel_classes"][split_id]
    )


def load_filtered_voc_instance(name, dirname, split, classnames):
    is_shot = "shot" in name
    dicts = []
    if is_shot:
        fileids = {}
        split_dir = os.path.join("datasets", "vocsplit")
        shot = name.split("_")[-2].split("shot")[0]
        seed = int(name.split("_seed")[-1])
        split_dir = os.path.join(split_dir, "seed_{}".format(seed))
        for cls in classnames:
            with PathManager.open(os.path.join(split_dir, "box_{}shot_{}_train.txt".format(shot, cls))) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
                ]
                fileids[cls] = fileids_
        for cls, fileids_ in fileids.items():
            dicts_ = []
            for fileid in fileids_:
                year = "2012" if "_" in fileid else "2007"
                dirname = os.path.join("datasets", "VOC{}".format(year))
                anno_file = os.path.join(
                    dirname, "Annotations", fileid + ".xml"
                )
                jpeg_file = os.path.join(
                    dirname, "JPEGImages", fileid + ".jpg"
                )

                tree = ET.parse(anno_file)

                for obj in tree.findall("object"):
                    r = {
                        "file_name": jpeg_file,
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                    }
                    cls_ = obj.find("name").text
                    if cls != cls_:
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances = [
                        {
                            "category_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    ]
                    r["annotations"] = instances
                    dicts_.append(r)
            if len(dicts_) > int(shot):
                dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        with PathManager.open(dirname, "ImageSets", "Main", split+".txt") as f:
            file_ids = np.loadtxt(f, dtype=np.str)
            for file_id in file_ids:
                anno_file = os.path.join(dirname, "Annotations", file_id+".xml")
                img_file = os.path.join(dirname, "JPEGImages", file_id+".jpg")

                tree = ET.parse(anno_file)
                instances = []
                r = {
                    "file_name": img_file,
                    "img_id": file_id,
                    "height": int(tree.findall("./size/height")[0].text),
                    "width": int(tree.findall("./size/width")[0].text)
                }
                for obj in tree.findall("object"):
                    cls = obj.find("name").text
                    if not (cls in classnames): continue
                    bbox = obj.find("bndbox")
                    bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances.append(
                        {
                            "cate_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS
                        }
                    )
                r["annotations"] = instances
                dicts.append(r)
    return dicts