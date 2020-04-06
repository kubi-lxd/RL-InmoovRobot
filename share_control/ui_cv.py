"""Script create user interface based on opencv"""
import cv2
import argparse
import numpy as np
from environments.inmoov.joints_registry import joint_info

BAR_LOW, BAR_HIGH = 0, 100
JOINT_NAME = [n[1] for n in joint_info]


class Slider:
    def __init__(self, win_name, dimension, sample_freq=10, slider_name=None, low=None, high=None):
        """

        :param win_name: the name for the slider pop up window
        :param dimension: total number of sliders
        :param sample_freq: the sample frequency from the slider
        :param slider_name: the name for every slider, it should be as long as the dimension
        :param low: the predefined lowest limit for the control object
        :param high: see above
        """
        self.win_name = win_name
        self.dim = dimension
        self.low = low
        self.high = high
        self.freq = sample_freq
        self.slider_low, self.slider_up = 0, 100
        self.ab = self.joint_info_process()
        if slider_name is None:
            self.slider_name = [str(i) for i in range(dimension)]
        else:
            self.slider_name = slider_name
        self.create_slider()

    def joint_info_process(self):
        """
        Return the "a", "b" in the equation a * low + b = 0; and a*high+b = 100
        For the moment, it is only suitable for inmoov case,
        In general this function is to make an affine change of variable, to project the joint low limit and high limit
        to the Interval [0, 100] for the further implementation convenience by OpenCV
        :return:
        """
        if self.low is None:
            ab = np.ones(shape=[self.dim, 2])
            ab[:, 1] = 0
            return ab
        joint_low_limit, joint_high_limit = self.low, self.high
        ab = []
        bar_low_high = np.array([[self.slider_low], [self.slider_up]])
        for i in range(len(joint_low_limit)):
            local_matrix = np.array([[joint_low_limit[i], 1],
                                    [joint_high_limit[i], 1]])
            inv_local = np.linalg.inv(local_matrix)
            local_ab = np.matmul(inv_local, bar_low_high)
            ab.append(local_ab.squeeze())
        ab = np.array(ab)
        return ab

    def slider_data_post_proc(self, data):
        """
        change the data get from the slider into the action space
        :param data: the data collected from slider
        :return:
        """
        a, b = self.ab[:, 0], self.ab[:, 1]
        data = (data - b) / a
        return data

    def create_slider(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(self.win_name, 500, 500)
        init_value = self.ab[:, 1]
        for i in range(self.dim):
            cv2.createTrackbar(self.slider_name[i], self.win_name, int(init_value[i]), 100, (lambda x: None))

    def get_slider_data(self):
        slider_info = []
        k = cv2.waitKey(self.freq) & 0xFF
        if k == 27:
            return
        for k in self.slider_name:
            slider_info.append(cv2.getTrackbarPos(k, self.win_name))
        if cv2.getWindowProperty(self.win_name, 0) < 0:
            return
        data = self.slider_data_post_proc(np.array(slider_info))
        data = data.astype(float)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create slider object User Interface")
    parser.add_argument('--name', type=str, default="inmoov")
    parser.add_argument('--freq', type=int, default=100,
                        help="Sample frequency for the slider object, get information from the slider")
    args, unknown = parser.parse_known_args()

    fig_name = "slider for {}".format(args.name)
    joint_info = np.array([p[:1] + p[2:] for p in joint_info])
    low, high = joint_info[:, 1], joint_info[:, 2]
    dimension = len(low)
    slider_name = JOINT_NAME
    # create a slider window, this window will pose automatically to collect the next data from the slider bar
    slider = Slider(win_name=fig_name, dimension=dimension,
                    sample_freq=args.freq, slider_name=slider_name, low=low, high=high)
    step = 0
    old_data = None
    data_new = None
    while True:
        print("global step {}".format(step))
        step += 1
        old_data = data_new
        data_new = slider.get_slider_data()
        if data_new is not None and old_data is not None:
            for j, e in enumerate(data_new):
                if e != old_data[j]:
                    print("Change at {}, from {:.2f} to {:.2f}".format(JOINT_NAME[j], old_data[j], data_new[j]))
        if data_new is None:
            break
    cv2.destroyAllWindows()
