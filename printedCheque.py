#!/usr/bin/env python
# coding: utf-8

# In[7]:

def printedRecog():
    import os
    import re
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'
    
    
    # In[8]:
    
    
    paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
        'IMAGE_PATH': os.path.join('images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
        'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
        'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
        'PROTOC_PATH':os.path.join('Tensorflow','protoc')
     }
    
    
    # In[9]:
    
    
    files = {
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }
    
    
    # In[10]:
    
    
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
    
    
    # In[11]:
    
    
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()
    
    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
    
    
    # In[12]:
    
    
    import cv2 
    import numpy as np
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    
    # In[13]:
    
    
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'],'printed.jpg')
    
    
    # In[14]:
    
    
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    
    plt.imshow(image_np_with_detections)
    plt.show()
    
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
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()
    
    
    # In[15]:
    
    
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
      
      Image_label=[]
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
      ROI_Dictionary={}
      counter_for = 0
      for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        height, width, channels = image.shape
        ymin = int(ymin*height)
        ymax = int(ymax*height)
        xmin = int(xmin*width)
        xmax = int(xmax*width)
        
        ROI_Dictionary[Image_label[counter_for]] = [ymin, ymax, xmin, xmax]
        coordinates_list.append([ymin, ymax, xmin, xmax, (box_to_score_map[box]*100)])
        counter_for = counter_for + 1
        
      #print(ROI_Dictionary)
      #print(ROI_Dictionary["date"])
      return ROI_Dictionary
    
    
    # In[16]:
    
    
    coordinates = return_coordinates(
    image_np_with_detections,
    np.squeeze(detections['detection_boxes']),
    np.squeeze(detections['detection_classes']+label_id_offset).astype(np.int32),
    np.squeeze(detections['detection_scores']),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.7)
    
    
    # In[17]:
    
    
    print(coordinates)
    
    
    # In[18]:
    
    
    import pytesseract
    from pytesseract import Output
    import requests
    from io import BytesIO
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tessdata_dir_config = r'--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'
    ExtractionResults={}
    
    
    # In[32]:
    
    
    def getPrintedText(ROI):
        text = pytesseract.image_to_string(ROI)
        return text
    
    def getMICR(PATH):
        
        imag = Image.open(PATH)
        width, height = imag.size
        img = imag.convert('LA')
        
        #convert to color code and crop image to only mcir
        test_img = cv2.cvtColor(np.array(imag), cv2.COLOR_BGR2GRAY)
        mcr_image = test_img[335:390, 45:500]
    
        plt.imshow(mcr_image)
        plt.show()
    
        #read string values and convert to mcr
        test = pytesseract.image_to_string(mcr_image, lang='mcr', config=tessdata_dir_config)
        
        return test[1:13]
    
    
    # In[33]:
    
    
    try:
        bankName = coordinates['digit']
        #Bank Name
        x= int(bankName[2])
        y= int(bankName[0])
        w= int(bankName[3])
        h= int(bankName[1])
        roi = image_np[y:h, x:w]
        
        ExtractionResults["digit"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff-\n]', ' ', getPrintedText(roi))
    
        plt.imshow(roi)
        plt.show()
            
    except:
        print ("Coordinates not Found")
    
    
    # In[34]:
    
    
    try:
        bankName = coordinates['logo']
        #Bank Name
        x= int(bankName[2])
        y= int(bankName[0])
        w= int(bankName[3])
        h= int(bankName[1])
        roi = image_np[y:h, x:w]
    
        ExtractionResults["logo"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff-\n]', ' ', getPrintedText(roi))
        
        plt.imshow(roi)
        plt.show()
    except:
        print ("Coordinates not Found")
    
    
    # In[35]:
    
    
    try:
        bankName = coordinates['amount']
        #Bank Name
        x= int(bankName[2])
        y= int(bankName[0])
        w= int(bankName[3])
        h= int(bankName[1])
        roi = image_np[y:h, x:w]
        
        ExtractionResults["amount"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff-\n]', ' ', getPrintedText(roi))
        
        plt.imshow(roi)
        plt.show()
    
    except:
        print ("Coordinates not Found")
    
    
    # In[36]:
    
    
    try:
        bankName = coordinates['address']
        #Bank Name
        x= int(bankName[2])
        y= int(bankName[0])
        w= int(bankName[3])
        h= int(bankName[1])
        roi = image_np[y:h, x:w]
    
        ExtractionResults["Address"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff-\n]', ' ', getPrintedText(roi))
        
        plt.imshow(roi)
        plt.show()
    except:
        print ("Coordinates not Found")
    
    
    # In[37]:
    
    
    try:
        ChequeNumber = getMICR(IMAGE_PATH)
        ExtractionResults["micr"] = ChequeNumber
    except:
        print ("Coordinates not Found")
    
    
    # In[40]:
    
    
    ChequeNumber = getMICR(IMAGE_PATH)
    print(ChequeNumber)
    
    
    # In[41]:
    
    
    try:
        bankName = coordinates['date']
        #Bank Name
        x= int(bankName[2])
        y= int(bankName[0])
        w= int(bankName[3])
        h= int(bankName[1])
        roi = image_np[y:h, x:w]
        
        ExtractionResults["date"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff-\n]', ' ', getPrintedText(roi))
    
        plt.imshow(roi)
        plt.show()
            
    except:
        print ("Coordinates not Found")
    
    
    # In[42]:
    
    
    try:
        bankName = coordinates['name']
        #Bank Name
        x= int(bankName[2])
        y= int(bankName[0])
        w= int(bankName[3])
        h= int(bankName[1])
        roi = image_np[y:h, x:w]
        
        ExtractionResults["name"] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff-\n]', ' ', getPrintedText(roi))
    
        plt.imshow(roi)
        plt.show()
            
    except:
        print ("Coordinates not Found")
    
    
   #%%
    
    global Plogo 
    
    if 'logo' in ExtractionResults:
        Plogo = ExtractionResults["logo"]
    else:
        Plogo = " "
    
    
    global Paddress
    
    if 'Address' in ExtractionResults:
        Paddress = ExtractionResults["Address"]
    else:
        Paddress = " "
        
    global Pdate
    
    if 'date' in ExtractionResults:
        Pdate = ExtractionResults["date"]
    else:
        Pdate = " "
        
    global Pamount
    
    if 'amount' in ExtractionResults:
        Pamount = ExtractionResults["amount"]
    else:
        Pamount = " "
        
    global Pdigit
    Pdigit = ExtractionResults["digit"]
    global Pname  
    
    if 'name' in ExtractionResults:
        Pname = ExtractionResults["name"]
    else:
        Pname = " "   
    
    global Pmcr
    
    if 'micr' in ExtractionResults:
        Pmcr = ExtractionResults["micr"]
    else:
        Pmcr = " "
    
    #%%
    print("----------------------")
    try:
        print("Logo: ", ExtractionResults["logo"]);
    except:
        print("logo not Found")
    print("----------------------")
    
    try:
        print("Address: ", ExtractionResults["Address"]);
    except:
        print("Address not Found")
    print("----------------------")
    
    try:        
        print("Date: ", ExtractionResults["date"]);
    except:
        print("date not found")
    print("----------------------")
    
    try:
        print("Amount: ", ExtractionResults["amount"]);
    except:
        print("Amount not found")
    print("----------------------")
    
    try:
        print("Digit: ", ExtractionResults["digit"]);
    except:
        print("digit not found")
    print("----------------------")
    
    try:
        print("Name: ", ExtractionResults["name"]);
    except:
        print("name not found")
    print("----------------------")
    
    try:
        print("MICR: ", ExtractionResults["micr"]);
    except:
        print("mcr not found")





