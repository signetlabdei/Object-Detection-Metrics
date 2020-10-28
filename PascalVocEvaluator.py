###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

import shutil
import glob

from Evaluator import *
from .pascalvoc import ValidateCoordinatesTypes, ValidateFormats, ValidateMandatoryArgs, \
    ValidateImageSize, getBoundingBoxes


class PascalVocEvaluator:

    def __init__(self, gtFolder=None, detFolder=None, iouThreshold=0.5,
                 gtFormat='xyrb', detFormat='xyrb', gtCoordinates='abs', detCoordinates='abs',
                 savePath=None, imgSize=(), savepath='', showPlot=False):
        # Get current path to set default folders
        self.currentPath = os.path.dirname(os.path.abspath(__file__))
        self.tmp_path = os.path.join(self.currentPath, 'tmp_files')
        self.VERSION = '0.1 (beta)'

        self.iouThreshold = iouThreshold

        # Arguments validation
        errors = []
        # Validate formats: ('xywh': <left> <top> <width> <height>) or ('xyrb': <left> <top> <right> <bottom>)
        self.gtFormat = ValidateFormats(gtFormat, '-gtformat', errors)
        self.detFormat = ValidateFormats(detFormat, '-detformat', errors)

        # temporary folders
        if gtFolder is None:
            if not os.path.exists(self.tmp_path):
                os.mkdir(self.tmp_path)
            # Groundtruth folder
            self.gtFolder = os.path.join(self.tmp_path, 'gt')
            os.mkdir(self.gtFolder)
        if detFolder is None:
            if not os.path.exists(self.tmp_path):
                os.mkdir(self.tmp_path)
            # Detection folder
            self.detFolder = os.path.join(self.tmp_path, 'det')
            os.mkdir(self.detFolder)

        # Coordinates types
        self.gtCoordType = ValidateCoordinatesTypes(gtCoordinates, '-gtCoordinates', errors)
        self.detCoordType = ValidateCoordinatesTypes(detCoordinates, '-detCoordinates', errors)
        # image size
        self.imgSize = (0, 0)
        if self.gtCoordType == CoordinatesType.Relative:  # Image size is required
            self.imgSize = ValidateImageSize(imgSize, '-imgsize', '-gtCoordinates', errors)
        if self.detCoordType == CoordinatesType.Relative:  # Image size is required
            self.imgSize = ValidateImageSize(imgSize, '-imgsize', '-detCoordinates', errors)

        if savePath is not None:
            self.savePath = self.ValidatePaths(savePath, '-sp/--savepath', errors)
        else:
            self.savePath = os.path.join(self.currentPath, 'results')
        # Validate savePath
        # If error, show error messages
        if len(errors) != 0:
            print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                                [-detformat] [-save]""")
            print('Object Detection Metrics: error(s): ')
            [print(e) for e in errors]
            sys.exit()

        # Check if path to save results already exists and is not empty
        if os.path.isdir(self.savePath) and os.listdir(self.savePath):
            key_pressed = ''
            while key_pressed.upper() not in ['Y', 'N']:
                print(f'Folder {self.savePath} already exists and may contain important results.\n')
                print(f'Enter \'Y\' to continue. WARNING: THIS WILL REMOVE ALL THE CONTENTS OF THE FOLDER!')
                print(f'Or enter \'N\' to abort and choose another folder to save the results.')
                key_pressed = input('')

            if key_pressed.upper() == 'N':
                print('Process canceled')
                sys.exit()

        # Clear folder and save results
        shutil.rmtree(self.savePath, ignore_errors=True)
        os.makedirs(self.savePath)
        # Show plot during execution
        self.showPlot = showPlot

    def _write_file(self, out_boxes, out_classes, out_scores=None, name=None):
        if not self.tmp_dir:
            self.tmp_dir = os.path.join(os.path.abspath(__file__ + "/../../"), 'tmp_results')
        classes_file = '../coco_classes.txt'

        is_gt = out_scores is None
        with open(classes_file, 'r') as rf:
            class_dict = {str(i): c.strip() for i, c in enumerate(rf.readlines())}
        if is_gt:
            file_out_path = os.path.join(self.gtFolder, name) if name else os.path.join(self.gtFolder,
                                                                                             'tmp.txt')
        else:
            file_out_path = os.path.join(self.detFolder, name) if name else os.path.join(self.detFolder,
                                                                                              'tmp.txt')

        with open(file_out_path, 'w') as fw:
            for n in range(len(out_classes)):
                if is_gt:
                    fw.write('{} {} {} {} {} '.format(class_dict[out_classes[n]],
                                                      int(out_boxes[n][1]),
                                                      int(out_boxes[n][0]),
                                                      int(out_boxes[n][3]),
                                                      int(out_boxes[n][2])))
                else:
                    fw.write('{} {} {} {} {} {} '.format(class_dict[out_classes[n]],
                                                         out_scores[n],
                                                         int(out_boxes[n][1]),
                                                         int(out_boxes[n][0]),
                                                         int(out_boxes[n][3]),
                                                         int(out_boxes[n][2])))
                fw.write('\n')
        return file_out_path

    def compute_score(self, det_bb, det_classes, det_scores,
                      gt_bb, gt_classes,
                      file_id):
        self._write_file(det_bb, det_classes, det_scores, name=file_id)
        self._write_file(gt_bb, gt_classes, det_scores, name=file_id)

        # Get groundtruth boxes
        allBoundingBoxes, allClasses = getBoundingBoxes(
            self.gtFolder, True, self.gtFormat, self.gtCoordType, imgSize=self.imgSize)
        # Get detected boxes
        allBoundingBoxes, allClasses = getBoundingBoxes(
            self.detFolder, False, self.detFormat, self.detCoordType, allBoundingBoxes, allClasses,
            imgSize=self.imgSize)
        allClasses.sort()

        evaluator = Evaluator()
        acc_AP = 0
        validClasses = 0

        # Plot Precision x Recall curve
        detections = evaluator.PlotPrecisionRecallCurve(
            allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=self.iouThreshold,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation,
            showAP=True,  # Show Average Precision in the title of the plot
            showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
            savePath=self.savePath,
            showGraphic=self.showPlot)

        f = open(os.path.join(self.savePath, 'results.txt'), 'w')
        f.write('Object Detection Metrics\n')
        f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
        f.write('Average Precision (AP), Precision and Recall per class:')

        # each detection is a class
        for metricsPerClass in detections:

            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']
            total_TP = metricsPerClass['total TP']
            total_FP = metricsPerClass['total FP']

            if totalPositives > 0:
                validClasses = validClasses + 1
                acc_AP = acc_AP + ap
                prec = ['%.2f' % p for p in precision]
                rec = ['%.2f' % r for r in recall]
                ap_str = "{0:.2f}%".format(ap * 100)
                # ap_str = "{0:.4f}%".format(ap * 100)
                print('AP: %s (%s)' % (ap_str, cl))
                f.write('\n\nClass: %s' % cl)
                f.write('\nAP: %s' % ap_str)
                f.write('\nPrecision: %s' % prec)
                f.write('\nRecall: %s' % rec)

        mAP = acc_AP / validClasses
        mAP_str = "{0:.2f}%".format(mAP * 100)
        print('mAP: %s' % mAP_str)
        f.write('\n\n\nmAP: %s' % mAP_str)

        self._reset_folders()
        return mAP

    def _reset_folders(self):
        gt_files = glob.glob(os.path.join(self.gtFolder, '*.txt'))
        for f in gt_files:
            os.remove(f)
        det_files = glob.glob(os.path.join(self.detFolder, '*.txt'))
        for f in det_files:
            os.remove(f)

    def ValidatePaths(self, arg, nameArg, errors):
        if arg is None:
            errors.append('argument %s: invalid directory' % nameArg)
        elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(self.currentPath, arg)) is False:
            errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
        # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
        #     arg = os.path.join(currentPath, arg)
        else:
            arg = os.path.join(self.currentPath, arg)
        return arg
