from xml.dom import minidom
from DataReader import Annotation
import os
from os import path as pth
import numpy as np
import shutil
class XML2JSON():

    def __read__(self , path):
        pass
            
    
    def __init__(self , xml_path):
        pass


class JSON2XML():
    """[JSON to XML Annotation]
    """
    
    def __read__(self, path):
        self.annotation = Annotation(path)

    def __init__(self, path : str = None , anot : Annotation = None):
        """[Note that one of two arguments "path" and "anot" must be passed to the initializer.]

        Args:
            path (str, optional): [Path of the JSON annotation]. Defaults to None.
            anot (Annotation, optional): [Annotaton object itself]. Defaults to None.
        """
        assert path is not None or anot is not None, 'No Valid Argument Passed'
       
        if path is not None:
            self.__read__(path)

        elif anot is not None:
            self.annotation = anot


    def getXML(self, use_convrtMaskToBbox = False):
        """[Returns the XML representation of a JSON annotation for BBOX]
        """
        filename = self.annotation.get_fname()
        folder = pth.split(self.annotation.get_path())[-1]
        full_path = pth.abspath(self.annotation.get_fullpath()) 
        size_width , size_height = self.annotation.get_img_size()
        
        if self.annotation.is_color():
            size_depth = 3
        else:
            size_depth = 1

        if use_convrtMaskToBbox:
            bounding_boxes = self.annotation.convert_mask_to_bbox() ## List of ((x_min , y_min) , (x_max , y_max))
        
        else:
            bounding_boxes = self.annotation.get_bboxs() ## List of ((x_min , y_min) , (x_max , y_max))
        
        
        root = minidom.Document()
        xml = root.createElement('annotation')
        xml = root.appendChild(xml)

        xml_folder = root.createElement('folder')
        xml_folder.appendChild(root.createTextNode(folder))
        
        xml_filename = root.createElement('filename')
        xml_filename.appendChild(root.createTextNode(filename))
        
        xml_path = root.createElement('folder')
        xml_path.appendChild(root.createTextNode(full_path))
        
        xml_source = root.createElement('source')
        xml_source.appendChild(
            root.createElement('database').appendChild(
                root.createTextNode('Unkown')
            )
        )

        xml_size_width = root.createElement('width')
        xml_size_width.appendChild(root.createTextNode(str(size_width)))
        xml_size_height = root.createElement('height')
        xml_size_height.appendChild(root.createTextNode(str(size_height)))
        xml_size_depth = root.createElement('depth')
        xml_size_depth.appendChild(root.createTextNode(str(size_depth)))

        xml_size = root.createElement('size')
        xml_size.appendChild(xml_size_width)
        xml_size.appendChild(xml_size_height)
        xml_size.appendChild(xml_size_depth)

        xml_segment = root.createElement('segmented')
        xml_segment.appendChild( root.createTextNode('0') )

        xml.appendChild(xml_folder)
        xml.appendChild(xml_filename)
        xml.appendChild(xml_path)
        xml.appendChild(xml_source)
        xml.appendChild(xml_size)
        xml.appendChild(xml_segment)

        for bbox in bounding_boxes:

            bbox_name = root.createElement('name')
            bbox_name.appendChild(
                root.createTextNode(str(bbox.get_class_id()))
            )

            bbox_pose = root.createElement('pose')
            bbox_pose.appendChild(
                root.createTextNode(bbox.get_pose())
            )

            bbox_truncated = root.createElement('truncated')
            bbox_truncated.appendChild(
                root.createTextNode(bbox.get_truncated())
            )

            bbox_difficult = root.createElement('difficult')
            bbox_difficult.appendChild(
                root.createTextNode(str(bbox.get_difficult()))
            )
            
            bndbox = root.createElement('bndbox')
            x_min = root.createElement('xmin')
            x_min.appendChild(
                root.createTextNode(str(bbox.get_bbox()[0])))

            y_min = root.createElement('ymin')
            y_min.appendChild(
                root.createTextNode(str(bbox.get_bbox()[1])))

            x_max = root.createElement('xmax')
            x_max.appendChild(
                root.createTextNode(str(bbox.get_bbox()[2])))

            y_max = root.createElement('ymax')
            y_max.appendChild(
                root.createTextNode(str(bbox.get_bbox()[3])))

            bndbox.appendChild(x_min)
            bndbox.appendChild(y_min)
            bndbox.appendChild(x_max)
            bndbox.appendChild(y_max)

            obj = root.createElement('object')
            obj.appendChild(bbox_name)
            obj.appendChild(bbox_pose)
            obj.appendChild(bbox_truncated)
            obj.appendChild(bbox_difficult)
            obj.appendChild(bndbox)
            
            xml.appendChild(obj)

        return root


    def saveXML(self, path, use_convrtMaskToBbox = False):
        """[Saves the JSON file as an XML annotation]

        Args:
            path ([str]): [Path for saving the file]
        """
        with open(path , 'w') as file:
            xml = self.getXML(use_convrtMaskToBbox = use_convrtMaskToBbox)
            xml_str = xml.toprettyxml(indent ="\t") 
            file.write(xml_str)
            file.close()



#______________________________________________________________________________________________________________________________________________
#explain:
#   ->A function which creates a progress bar in console, full documentation in the link bellow. Check out the comments in the
#       code itself.
#       link: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
#______________________________________________________________________________________________________________________________________________
def __printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\b"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()





def create_dataset_XML(image_path:str , annotation_path : str , save_path_img : str, save_path_anotation:str, copyImages = False):
    """Using the input parameters, this function will read the annotation of the images, convert them to XML format, saves them
    in annotation save path and copy the coresponding images to image save path.

    Args:
        image_path (str): [path of the images]
        annotation_path (str): [path of JSON annotations]
        save_path_img (str): [path where the images will be copied]
        save_path_anotation (str): [path where XML annotations are saved]
    """
    file_names = os.listdir(annotation_path)

    total_annotations = len(file_names)
    print('Creating Object Detection Dataset!\nPlease Wait...')
    for idx, file_name in enumerate(file_names):
        
        full_path = pth.join(annotation_path , file_name)
        file_name_xml = pth.splitext(file_name)[0] + '.xml'
        file_name_jpg = pth.splitext(file_name)[0] + '.jpg'

        anot = Annotation(path=full_path)

        if(anot.have_object()):
            JSON2XML(anot = anot).saveXML(pth.join(save_path_anotation , file_name_xml) , use_convrtMaskToBbox=True)
            if copyImages:
                shutil.copyfile(pth.join(image_path,file_name_jpg), pth.join(save_path_img,file_name_jpg))


        __printProgressBar(idx +1, total_annotations,prefix='Converting/Copying' , suffix='Completed')
    print('\n...\nObject Detection Dataset Created Successfully.\nPress any key to continue...')
    input()

def main():
    os.system('clear')
    create_dataset_XML(
        image_path = r'severstal-steel-defect-detection\train_images',
        annotation_path = r'severstal-steel-defect-detection\annotations_json',
        save_path_img = r'severstal-steel-defect-detection\od-dataset\images',
        save_path_anotation = r'severstal-steel-defect-detection\od-dataset\annotations',
        copyImages = False
        )


if __name__ == '__main__':
    main()