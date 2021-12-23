def test():    
    import os
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'
    
    
    # In[2]:
    
    
    paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
        'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
        'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
        'PROTOC_PATH':os.path.join('Tensorflow','protoc')
     }
    
    
    # In[3]:
    
    
    files = {
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }
    
    
    # In[4]:
    
    
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
    
    
    # In[5]:
    
    
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
    
    
    # In[6]:
    
    
    
    import cv2 
    import numpy as np
    from matplotlib import pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    
    # In[7]:
    
    Image_To_Read ="C:/Users/Neha/TFODCourse-main/Tensorflow/workspace/images/test/1635145198344.jpg"
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', Image_To_Read)
    
    
    # In[8]:
    
    
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    #print(detections)
    #plt.imshow(detections)
    #plt.show()
    
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
    
    
    # In[9]:
    
    
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
    
    
    # In[10]:
    
    
    coordinates = return_coordinates(
    image_np_with_detections,
    np.squeeze(detections['detection_boxes']),
    np.squeeze(detections['detection_classes']+label_id_offset).astype(np.int32),
    np.squeeze(detections['detection_scores']),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.7)
    
    
    # In[11]:
    
    
    print(coordinates)
    
    
    # In[12]:
    
    
    import pytesseract
    from pytesseract import Output
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    
    # In[13]:
    
    
    
    # bankName = coordinates['name']
    # #Bank Name
    # x= int(bankName[2])
    # y= int(bankName[0])
    # w= int(bankName[3])
    # h= int(bankName[1])
    # roi = image_np[y:h, x:w]
    
    # plt.imshow(roi)
    # plt.show()
    
    
    # In[14]:
    
    
    # bankName = coordinates['date']
    # #Bank Name
    # x= int(bankName[2])
    # y= int(bankName[0])
    # w= int(bankName[3])
    # h= int(bankName[1])
    # roi = image_np[y:h, x:w]
    
    # plt.imshow(roi)
    # plt.show()
    
    
    # In[15]:
    
    
    # bankName = coordinates['logo']
    # #Bank Name
    # x= int(bankName[2])
    # y= int(bankName[0])
    # w= int(bankName[3])
    # h= int(bankName[1])
    # roi = image_np[y:h, x:w]
    
    # plt.imshow(roi)
    # plt.show()
    
    
    # In[16]:
    
    
    # try:
    #     bankName = coordinates['mcir']
    #     #Bank Name
    #     x= int(bankName[2])
    #     y= int(bankName[0])
    #     w= int(bankName[3])
    #     h= int(bankName[1])
    #     roi = image_np[y:h, x:w]
    
    
    
    #     plt.imshow(roi)
    #     plt.show()
    # except:
    #     print ("Coordinates not Found")
    
    
    # In[17]:
    
    
    # try:
    #     bankName = coordinates['address']
    #     #Bank Name
    #     x= int(bankName[2])
    #     y= int(bankName[0])
    #     w= int(bankName[3])
    #     h= int(bankName[1])
    #     roi = image_np[y:h, x:w]
    
    #     plt.imshow(roi)
    #     plt.show()
    # except:
    #     print ("Coordinates not Found")
    
    
    # In[18]:
    
    
    #for i in coordinates:
    #    x= int(i[2])
    #    y= int(i[0])
    #    w= int(i[3])
    #    h= int(i[1])
        
    #    roi = image_np[y:h, x:w]
    
    #    plt.imshow(roi)
    #    plt.show()
        
    
    
    # In[19]:
    
    
    import cv2
    import numpy as np
    import pandas as pd 
    
    
    # In[20]:
    
    
    from PIL import Image 
    
    
    # In[21]:
    
    
    # pip install matplotlib opencv-python 
    
    
    # In[22]:
    
    
    import matplotlib.pyplot as plt
    
    
    # In[23]:
    
    
    # digit = coordinates['date']
    # #Bank Name
    # x= int(digit[2])
    # y= int(digit[0])
    # w= int(digit[3])
    # h= int(digit[1])
    # digit_original = image_np[y:h, x:w]
    
    # plt.imshow(digit_original)
    # plt.show()
    
    
    # In[24]:
    
    
    try:
        digit = coordinates['digit']
        x= int(digit[2])
        y= int(digit[0])
        w= int(digit[3])
        h= int(digit[1])
        digit_original = image_np[y:h, x:w]
        digit_roi = image_np[y:h, x:w]
        
        
        plt.imshow(digit_original)
        plt.show()
         
        
    except:
        print ("Coordinates not Found")
    
    
    # In[25]:
    
    
    #pip install --user  opencv-contrib-python
    
    
    # In[26]:
    
    
    src = digit_original
    
    # set a new height in pixels
    height  = 500
    width = 1200
    # dsize
    dsize = (width, height)
    
    # resize image
    output = cv2.resize(src, dsize)
    
    
    # In[27]:
    
    
    import os
    #delete temp folder + all its content
    directory = "temp"
    parent_dir = "C:/Users/User/Desktop/yt_object_detection/"
    
    path = os.path.join(parent_dir, directory)
    
    try:
        import shutil
        shutil.rmtree(path)
    except Exception as e:
            print (e)
    
    
    # In[28]:
    
    
    
    try:
        directory = "temp"
        parent_dir = "C:/Users/User/Desktop/yt_object_detection/"
    
        path = os.path.join(parent_dir, directory)
    
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
    
        if not isExist:
            os.mkdir(path)
            print("Directory '% s' created" % directory)
        else:
            print("Directory '% s' already exists" % directory)
    except Exception as e:
            print (e)
    
    
    # In[77]:
    
    
    #crop each digit
    orig_image = output
    
    def process(image):
    	image= cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    	kernel = np.ones((5,5),np.uint8)
    	image = cv2.dilate(image,kernel,iterations = 1)
    	ret,thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
    	inv = cv2.bitwise_not(thresh)
    	struct = np.ones((3,3),np.uint8)
    	dilated = cv2.dilate(inv ,struct,iterations=1)	
    	edges = cv2.Canny(dilated,30,200)
    	return edges,dilated
    
    def manage_contours(image,orig_image):
    	results=[]
    	contours,hier = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    	test = cv2.drawContours(orig_image,contours,-1,(255,255,255),2)
    	cv2.imwrite("test.png",test)
    	for cnt in contours:
    		x,y,w,h = cv2.boundingRect(cnt)
    		small_image = orig_image[y:y+h,x:x+w]
    		results.append(small_image)
    	return orig_image,contours,results
    
    def save_and_process_individual_images(ilist):
    	for c,img in enumerate(ilist):
    		edg,dil = process(img)
    		cv2.imwrite('temp/'+ str(c)+".png",dil)
    
    def get_processed_images(ilist):
    	res = []
    	for img in ilist:
    		edg,dil = process(img)
    		res.append((edg,dil))
    	return res 
    
    	
    edges,dilated = process(orig_image)
    new_image,contours,res = manage_contours(edges,orig_image.copy())
    save_and_process_individual_images(res)
    
    
    # In[30]:
    
    
    # #delete temp folder + all its content
    # try:
    #     import shutil
    #     shutil.rmtree(path)
    # except Exception as e:
    #         print (e)
    
    
    # In[31]:
    
    
    from tensorflow.python.keras.models import load_model
    import matplotlib.pyplot as plt
    
    
    # In[ ]:
    
    
    
    
    
    # In[62]:
    
    predictionlist=[]
    def predict(img):
        image = img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]
        image = cv2.resize(image, (28, 28))
    #     display_image(image)
        image = image.astype('float32')
        image = image.reshape(1, 28, 28, 1)
        image /= 255
    
        plt.imshow(image.reshape(28, 28), cmap='Greys')
        plt.show()
        model = load_model('digits_recognition_cnn.h5')
        pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
        predictionlist.append(pred.argmax())
        
        #print("Predicted Number: ", pred.argmax())
        
    
    
    # In[63]:
    
    
    # predict(cv2.imread('temp/4.png'))
    
    
    # In[64]:
    
    
    filelist=os.listdir('temp')
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)
    #print(filelist)
    
    
    # In[7]:
    
    
    # In[65]:
    
    
    for x in filelist:
        predict(cv2.imread('temp/' + x))
    
    
    
    #predictionlist
    #print(*predictionlist, sep = "")
    
    digit = ''.join(str(e) for e in predictionlist)
    
    global x1
    x1 = digit
    
    """
    import json
    
    retJson = {
        'digit': digit
        }
    
    with open('text.json','w') as f:
        json.dump(retJson,f)
     """
    
test()