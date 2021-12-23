# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:37:36 2021

@author: Neha
"""


def recognition():
    import os
    import re
    import handDetection
    import PIL
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    #%%

    paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
        'IMAGE_PATH': os.path.join('images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
        'TFJS_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
        'TFLITE_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
        'PROTOC_PATH': os.path.join('Tensorflow', 'protoc')
    }
    
    print(paths)
    
#%%
    files = {
        'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }
#%%
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
#%%

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(
        files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(
        paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
#%%

    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
#%%

    category_index = label_map_util.create_category_index_from_labelmap(
        files['LABELMAP'])
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'],'temp.jpg')

#%%

    img = cv2.imread('C:\\TFODCourse-main\\images\\temp.jpg')
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}

    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    #plt.imshow(image_np_with_detections)
    #plt.show()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=8,
        min_score_thresh=.9,
        agnostic_mode=False)
    #cv2.imwrite(r"C:\Users\hp\Desktop\YT_OCR\ROI\test.jpg", image_np_with_detections)
    #plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    #plt.show()

    #%%

    import collections

    STANDARD_COLORS = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
        'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]

    def return_coordinates(
            image,
            boxes,
            classes,
            scores,
            category_index,
            instance_masks=None,
            instance_boundaries=None,
            keypoints=None,
            use_normalized_coordinates=False,
            max_boxes_to_draw=20,
            min_score_thresh=.5,
            agnostic_mode=False,
            line_thickness=4,
            groundtruth_box_visualization_color='black',
            skip_scores=False,
            skip_labels=False):

      Image_label = []
      # Create a display string (and color) for every box location, group any boxes
      # that correspond to the same location.
      box_to_display_str_map = collections.defaultdict(list)
      box_to_color_map = collections.defaultdict(str)
      box_to_instance_masks_map = {}
      box_to_instance_boundaries_map = {}
      box_to_score_map = {}
      box_to_keypoints_map = collections.defaultdict(list)
      if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
      for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
          box = tuple(boxes[i].tolist())
          if instance_masks is not None:
            box_to_instance_masks_map[box] = instance_masks[i]
          if instance_boundaries is not None:
            box_to_instance_boundaries_map[box] = instance_boundaries[i]
          if keypoints is not None:
            box_to_keypoints_map[box].extend(keypoints[i])
          if scores is None:
            box_to_color_map[box] = groundtruth_box_visualization_color
          else:
            display_str = ''
            if not skip_labels:
              if not agnostic_mode:
                if classes[i] in category_index.keys():
                  class_name = category_index[classes[i]]['name']
                  #print(class_name)
                  #print(i)
                  Image_label.append(class_name)
                else:
                  class_name = 'N/A'
                display_str = str(class_name)
            if not skip_scores:
              if not display_str:
                display_str = '{}%'.format(int(100*scores[i]))
              else:
                display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
            box_to_display_str_map[box].append(display_str)
            box_to_score_map[box] = scores[i]
            if agnostic_mode:
              box_to_color_map[box] = 'DarkOrange'
            else:
              box_to_color_map[box] = STANDARD_COLORS[
                  classes[i] % len(STANDARD_COLORS)]

      # Draw all boxes onto image.
      coordinates_list = []
      ROI_Dictionary = {}
      counter_for = 0
      for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        height, width, channels = image.shape
        ymin = int(ymin*height)
        ymax = int(ymax*height)
        xmin = int(xmin*width)
        xmax = int(xmax*width)

        ROI_Dictionary[Image_label[counter_for]] = [ymin, ymax, xmin, xmax]
        coordinates_list.append(
            [ymin, ymax, xmin, xmax, (box_to_score_map[box]*100)])
        counter_for = counter_for + 1

      #print(ROI_Dictionary)
      #print(ROI_Dictionary["date"])
      return ROI_Dictionary

#%%

    coordinates = return_coordinates(
        image_np_with_detections,
        np.squeeze(detections['detection_boxes']),
        np.squeeze(detections['detection_classes'] +
                   label_id_offset).astype(np.int32),
        np.squeeze(detections['detection_scores']),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.5)
    
    #print(coordinates)
    
    import pytesseract
    from pytesseract import Output
    import requests
    from io import BytesIO
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tessdata_dir_config = r'--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'
    ExtractionResults = {}

    #%%

    def getPrintedText(ROI):
        text = pytesseract.image_to_string(ROI)
        return text

    def getWord(ROI):
        img = ROI
        height, width, c = img.shape

        #letter_boxes = pytesseract.image_to_boxes(img)
        #print(letter_boxes)

        for box in letter_boxes.splitlines():
            box = box.split()
            #print(box)
            x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            cv2.rectangle(img, (x, height-y), (w, height-h), (0, 0, 255), 1)
            cv2.putText(img, box[0], (x, height-h+32),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        #plt.imshow(img)
        #plt.show()
        #cv2.imwrite('eeeeee.png', img)
        #cv2.imshow("window", img)
        #cv2.waitKey(0)

    def getMICR(PATH):

        imag = Image.open(PATH)
        width, height = imag.size
        img = imag.convert('LA')

        #convert to color code and crop image to only mcir
        test_img = cv2.cvtColor(np.array(imag), cv2.COLOR_BGR2GRAY)
        mcr_image = test_img[335:390, 45:500]

        #plt.imshow(mcr_image)
        #plt.show()

        #read string values and convert to mcr
        test = pytesseract.image_to_string(
            mcr_image, lang='micr', config=tessdata_dir_config)

        return test[1:13]

    try:
        bankName = coordinates['logo']
        #Bank Name
        x = int(bankName[2])
        y = int(bankName[0])
        w = int(bankName[3])
        h = int(bankName[1])
        roi = image_np[y:h, x:w]
        
        #print(getPrintedText(roi))
        
        ExtractionResults["logo"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff-\n]', ' ', getPrintedText(roi))
        
        #plt.imshow(roi)
        #plt.show()
    except:
        print("Coordinates not Found")

    try:
        bankName = coordinates['address']
        #Bank Name
        x = int(bankName[2])
        y = int(bankName[0])
        w = int(bankName[3])
        h = int(bankName[1])
        roi = image_np[y:h, x:w]

        ExtractionResults["Address"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff-\n]', ' ', getPrintedText(roi))

        #plt.imshow(roi)
        #plt.show()
    except:
        print("Coordinates not Found")
        
    try:
        bankName = coordinates['digit']
        #Bank Name
        x = int(bankName[2])
        y = int(bankName[0])
        w = int(bankName[3])
        h = int(bankName[1])
        
        roi = image_np[y:h, x:w]
        #print(roi)
        #plt.imshow(roi)
        #plt.show()
        image = PIL.Image.fromarray(roi, "RGB")
        #print(image)
        cv2.imwrite('C:\\TFODCourse-main\\images\\digit.jpg', roi)
        # plt.imshow(roi)
        # plt.show()
    except:
        print("Coordinates not Found")
        
    try:
        bankName = coordinates['date']
        #Bank Name
        x = int(bankName[2])
        y = int(bankName[0])
        w = int(bankName[3])
        h = int(bankName[1])
        
        roi = image_np[y:h, x:w]
        #print(roi)
        plt.imshow(roi)
        plt.show()
        image = PIL.Image.fromarray(roi, "RGB")
        #print(image)
        cv2.imwrite('C:\\TFODCourse-main\\TFODCourse-main\\images\\date.jpg', roi)
        # plt.imshow(roi)
        # plt.show()
    except:
        print("Coordinates not Found")
        
    try:
        ChequeNumber = getMICR(IMAGE_PATH)
        ExtractionResults["micr"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff-\n]', ' ', ChequeNumber)
    except:
        print("Coordinates not Found")


    if 'logo' in ExtractionResults:
        logo1 = ExtractionResults["logo"]
    else:
        logo1 = ""
    

    
    #print(ExtractionResults["logo"])
    global logoG
    logoG = logo1

    if 'Address' in ExtractionResults:
        Address1 = ExtractionResults["Address"]
    else:
        Address1 = ""

    #print(ExtractionResults["Address"])

    global AddressG
    AddressG = Address1
    
    if 'micr' in ExtractionResults:
        mcr1 = ExtractionResults["micr"]
    else:
        mcr1 = ""
    #print(ExtractionResults["micr"])
    global mcrG
    mcrG = mcr1




