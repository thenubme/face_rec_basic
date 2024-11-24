import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;

public class FaceRecognition {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    // Known faces database (embeddings and names)
    private static final Map<String, float[]> KNOWN_FACES = new HashMap<>();

    public static void main(String[] args) {
        // Load Haar Cascade for face detection
        String haarCascadePath = "data/haarcascade_frontalface_default.xml";
        CascadeClassifier faceDetector = new CascadeClassifier(haarCascadePath);

        if (faceDetector.empty()) {
            System.err.println("Error loading Haar Cascade file!");
            return;
        }

        // Load known faces
        loadKnownFaces("data/known_faces.csv");

        // Open webcam
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.err.println("Cannot open webcam!");
            return;
        }

        // Display frame
        JFrame frame = new JFrame("Real-Time Face Recognition");
        JLabel videoLabel = new JLabel();
        frame.add(videoLabel);
        frame.setSize(800, 600);
        frame.setVisible(true);

        Mat videoFrame = new Mat();
        MatOfRect faces = new MatOfRect();

        while (camera.read(videoFrame)) {
            // Convert frame to grayscale for detection
            Mat grayFrame = new Mat();
            Imgproc.cvtColor(videoFrame, grayFrame, Imgproc.COLOR_BGR2GRAY);

            // Detect faces
            faceDetector.detectMultiScale(grayFrame, faces);

            // Recognize each detected face
            for (Rect face : faces.toArray()) {
                Mat faceROI = videoFrame.submat(face);
                String name = recognizeFace(faceROI);

                // Draw rectangle and label
                Imgproc.rectangle(videoFrame, new Point(face.x, face.y),
                        new Point(face.x + face.width, face.y + face.height), new Scalar(0, 255, 0), 2);
                Imgproc.putText(videoFrame, name, new Point(face.x, face.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            }

            // Display frame in JFrame
            BufferedImage bufferedImage = matToBufferedImage(videoFrame);
            videoLabel.setIcon(new ImageIcon(bufferedImage));
            frame.repaint();

            if (!frame.isVisible()) break;
        }

        camera.release();
        System.exit(0);
    }

    private static void loadKnownFaces(String filePath) {
        // Load known faces and embeddings from CSV file
        // Format: Name, Embedding1, Embedding2,...,EmbeddingN
        // Example: John,0.12,0.34,0.56
        KNOWN_FACES.put("John Doe", new float[]{0.1f, 0.2f, 0.3f}); // Example data
    }

    private static String recognizeFace(Mat faceROI) {
        // Convert faceROI to model-compatible input (resize, normalize)
        Mat resizedFace = new Mat();
        Imgproc.resize(faceROI, resizedFace, new Size(160, 160));

        // Extract embeddings (dummy implementation here)
        float[] faceEmbedding = extractEmbedding(resizedFace);

        // Compare with known faces
        return findClosestMatch(faceEmbedding);
    }

    private static float[] extractEmbedding(Mat face) {
        // Replace with actual model inference code (e.g., Dlib, TensorFlow)
        return new float[]{0.1f, 0.2f, 0.3f}; // Dummy embedding
    }

    private static String findClosestMatch(float[] embedding) {
        double minDistance = Double.MAX_VALUE;
        String bestMatch = "Unknown";

        for (Map.Entry<String, float[]> entry : KNOWN_FACES.entrySet()) {
            double distance = calculateDistance(entry.getValue(), embedding);
            if (distance < minDistance) {
                minDistance = distance;
                bestMatch = entry.getKey();
            }
        }

        return bestMatch;
    }

    private static double calculateDistance(float[] a, float[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = mat.channels() > 1 ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;
        byte[] buffer = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, buffer);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), buffer);
        return image;
    }
}
