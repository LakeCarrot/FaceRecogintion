import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Created by bohu on 4/3/17.
 */
public class OpenCVFaceRecognizer {

    public static void main(String args[]) {
        //FaceTraining();
        FaceRecognition();
    }

    public static void FaceTraining() {
        String trainingDir = "../dataSet/train";
        File root = new File(trainingDir);
        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };
        File[] imageFiles = root.listFiles(imgFilter);
        System.out.println(imageFiles.length);
        opencv_core.MatVector images = new opencv_core.MatVector(imageFiles.length);
        opencv_core.Mat labels = new opencv_core.Mat(imageFiles.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();
        int counter = 0;
        for (File image : imageFiles) {
            opencv_core.Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int label = Integer.parseInt(image.getName().split("\\-")[0]);
            images.put(counter, img);
            labelsBuf.put(counter, label);
            counter++;
        }


        opencv_face.FaceRecognizer faceRecognizer = createFisherFaceRecognizer();
        faceRecognizer.train(images, labels);
        faceRecognizer.save("./FisherFaceRecognizer.yml");
    }

    public static void FaceRecognition() {

        String trainingDir = "../dataSet/test";
        File root = new File(trainingDir);
        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };
        File[] imageFiles = root.listFiles(imgFilter);
        opencv_face.FaceRecognizer faceRecognizer = createFisherFaceRecognizer();
        faceRecognizer.load("./FisherFaceRecognizer.yml");
        double sum = 0;
        double totalSize = 0;
        for (File image : imageFiles) {
            opencv_core.Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            long startTime = System.currentTimeMillis();
            int predictedLabel = faceRecognizer.predict(img);
            long endTime = System.currentTimeMillis();
            System.out.println("Label " + predictedLabel);
            //System.out.println("face recognizer " + "Total execution time: " + (endTime - startTime));
            sum += endTime - startTime;
            totalSize += image.length();
        }
        System.out.println(totalSize/imageFiles.length/1024);
        System.out.println(sum/imageFiles.length);
    }

}
