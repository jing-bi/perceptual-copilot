from PIL import Image
import base64
import supervision as sv
import numpy as np
import cv2
colors = sv.ColorPalette.from_hex(
    [
        "#a1c9f4",
        "#ffb482",
        "#8de5a1",
        "#ff9f9b",
        "#d0bbff",
        "#debb9b",
        "#fab0e4",
        "#cfcfcf",
        "#fffea3",
        "#b9f2f0",
        "#a1c9f4",
        "#ffb482",
        "#8de5a1",
        "#ff9f9b",
        "#d0bbff",
        "#debb9b",
        "#fab0e4",
        "#cfcfcf",
        "#fffea3",
        "#b9f2f0",
    ]
)

def image_w_box(image,objxbox):

    box_annotator = sv.BoxCornerAnnotator(thickness=10, corner_length=30, color=colors)
    label_annotator = sv.LabelAnnotator(color=colors)
    mask_annotator = sv.MaskAnnotator(opacity=0.2, color=colors)

    xyxys = np.array([v for boxes in objxbox.values() for v in boxes])
    unique_labels = sorted(objxbox.keys())
    class_id_map = dict(enumerate(unique_labels))
    labels = [l for l, boxes in objxbox.items() for _ in boxes]
    class_id = [list(class_id_map.values()).index(label) for label in labels]

    masks = np.zeros((len(xyxys), image.shape[0], image.shape[1]), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(xyxys):
        masks[i, int(y1):int(y2), int(x1):int(x2)] = labels[i]

    if len(xyxys) == 0:
        return image
    detections = sv.Detections(
        xyxy=xyxys,
        mask=masks,
        class_id=np.array(class_id),
    )
    # Convert RGB to BGR for annotation
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # After annotation, convert back to RGB
    annotated_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)

    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


def encode_image(img) -> tuple[str, str]:
    arr = np.array(img.convert("RGB")) if isinstance(img, Image.Image) else img
    if not isinstance(arr, np.ndarray):
        raise ValueError("Unsupported image type")
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Encoding failed")
    b64 = base64.b64encode(buf).decode('utf-8')
    return b64, "image/jpeg"