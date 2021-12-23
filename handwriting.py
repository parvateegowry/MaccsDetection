# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 08:48:45 2021

@author: Neha
"""
def handwritingDetection():
    import os
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'
    
    #%%
    paths = {
    'WORKSPACE_PATH': os.path.join('handwriting', 'workspace'),
    'SCRIPTS_PATH': os.path.join('handwriting','scripts'),
    'APIMODEL_PATH': os.path.join('handwriting','models'),
    'ANNOTATION_PATH': os.path.join('handwriting', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('handwriting', 'workspace','images'),
    'MODEL_PATH': os.path.join('handwriting', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('handwriting', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('handwriting', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('handwriting', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('handwriting', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('handwriting', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('handwriting','protoc')
    }
    
    #%%
    files = {
    'PIPELINE_CONFIG':os.path.join('handwriting', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }
    
    #%%
    
    import object_detection
    labels = [{'name':'one', 'id':1}, {'name':'two', 'id':2}, 
          {'name':'Three', 'id':3}, {'name':'Four', 'id':4}, 
          {'name':'five', 'id':5}, {'name':'six', 'id':6}, 
          {'name':'Seven', 'id':7}, {'name':'eight', 'id':8},
          {'name':'nine', 'id':9}, {'name':'Ten', 'id':10}, 
          {'name':'Twelve', 'id':11}, {'name':'Thirteen', 'id':12}, 
          {'name':'Fourteen', 'id':13}, {'name':'Seventeen', 'id':14}, 
          {'name':'Eighteen', 'id':15},{'name':'Nineteen', 'id':16}, 
          {'name':'twenty', 'id':17}, {'name':'Thirty', 'id':18}, 
          {'name':'Forty', 'id':19}, {'name':'Fifty', 'id':20}, 
          {'name':'Sixty', 'id':21}, {'name':'Seventy', 'id':22},
          {'name':'eighty', 'id':23}, {'name':'Ninety', 'id':24}, 
          {'name':'hundred', 'id':25}, {'name':'Hundred', 'id':26}, 
          {'name':'Thousand', 'id':27}, {'name':'and', 'id':28}, 
          {'name':'Cents', 'id':29}, {'name':'only', 'id':30}, 
          {'name':'Five', 'id':31}, {'name':'Six', 'id':32},
          {'name':'Eight', 'id':33}, {'name':'50', 'id':34},
          {'name':'75', 'id':35}, {'name':'25', 'id':36},
          {'name':'fifty', 'id':36}, {'name':'.', 'id':37},
          {'name':'Fifteen', 'id':38},{'name':'cents', 'id':39}]
    
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')
        #%%
        
    import tensorflow as tf
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format
    
    import os
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
    
    
    #%%
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-10')).expect_partial()
    
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
    
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', '1635145199111.jpg')
    
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
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=8,
                min_score_thresh=.7,
                agnostic_mode=False)
    
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()


handwritingDetection()
    
    
    