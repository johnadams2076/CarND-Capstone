from styx_msgs.msg import TrafficLight
from keras.models import load_model
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # Model downloaded from https://github.com/JunshengFu/traffic-light-classifier 
        model_path = rospy.get_param('~model_path')
        self.model=load_model(model_path)

        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_dim=(32,32)
        image_fin = cv2.resize(image, image_dim, interpolation=cv2.INTER_LINEAR)
        image = np.expand_dims(np.array(image_fin), axis=0)

        color_index = self.model.predict_classes(image)
        print("Color Index", color_index)
        return color_index
        #return TrafficLight.UNKNOWN
