import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;

public class FaceRecognition {
    // Load OpenCV native library
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    // Database of known faces (name and feature embeddings)
    private static final Map<String, float[]> KNOWN_FACES = new HashMap<>();

    public static void main(String[] args) {
        initializeKnownFaces(); // Load known faces into memory
        openCameraAndRecognizeFaces(); // Start real-time face recognition
    }

    /**
     * Initialize the known faces database with hardcoded embeddings.
     */
    private static void initializeKnownFaces() {
        KNOWN_FACES.put("Unknown", new float[]{
                // Example 128-dimensional embedding for "Himanshu"
                0.1f, 0.2f // Add complete 128 values here
        });
    }

    /**
     * Open webcam and perform real-time face recognition.
     */
    private static void openCameraAndRecognizeFaces() {
        // Load Haar Cascade file for face detection
        String haarCascadePath = "data/haarcascade_frontalface_default.xml";
        CascadeClassifier faceDetector = new CascadeClassifier(haarCascadePath);

        if (faceDetector.empty()) {
            System.err.println("Error: Haar Cascade file not found!");
            return;
        }

        // Open the default webcam
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.err.println("Error: Cannot open webcam!");
            return;
        }

        // Setup UI for displaying the video feed
        JFrame frame = new JFrame("Real-Time Face Recognition Testing , v1 by Himanshu");
        JLabel videoLabel = new JLabel();
        frame.add(videoLabel);
        frame.setSize(800, 600);
        frame.setVisible(true);

        Mat videoFrame = new Mat(); // Holds the current video frame
        MatOfRect faces = new MatOfRect(); // Stores detected face coordinates

        while (camera.read(videoFrame)) {
            // Convert the video frame to grayscale for better detection
            Mat grayFrame = new Mat();
            Imgproc.cvtColor(videoFrame, grayFrame, Imgproc.COLOR_BGR2GRAY);

            // Detect faces in the current frame
            faceDetector.detectMultiScale(grayFrame, faces);

            // Process each detected face
            for (Rect face : faces.toArray()) {
                Mat faceROI = videoFrame.submat(face); // Extract face region
                String name = recognizeFace(faceROI); // Identify the face

                // Draw a rectangle and label around the detected face
                Imgproc.rectangle(videoFrame, new Point(face.x, face.y),
                        new Point(face.x + face.width, face.y + face.height), new Scalar(0, 255, 0), 2);
                Imgproc.putText(videoFrame, name, new Point(face.x, face.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            }

            // Display the processed frame in the UI
            BufferedImage bufferedImage = matToBufferedImage(videoFrame);
            videoLabel.setIcon(new ImageIcon(bufferedImage));
            frame.repaint();

            // Exit loop if the window is closed
            if (!frame.isVisible()) break;
        }

        // Release camera resources and exit
        camera.release();
        System.exit(0);
    }

    /**
     * Recognize a face by comparing its embedding to the known database.
     */
    private static String recognizeFace(Mat faceROI) {
        // Generate placeholder embeddings for testing
        float[] faceEmbedding = new float[128];
        for (int i = 0; i < 128; i++) faceEmbedding[i] = 0.5f;

        // Compare the embedding with known faces
        double minDistance = Double.MAX_VALUE;
        String bestMatch = "Unknown";

        for (Map.Entry<String, float[]> entry : KNOWN_FACES.entrySet()) {
            double distance = calculateDistance(entry.getValue(), faceEmbedding);
            if (distance < minDistance) {
                minDistance = distance;
                bestMatch = entry.getKey();
            }
        }

        // Return "Unknown" if no match is within threshold
        return (minDistance > 1.0) ? "Unknown" : bestMatch;
    }

    /**
     * Calculate Euclidean distance between two 128-dimensional embeddings.
     */
    private static double calculateDistance(float[] a, float[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    /**
     * Convert OpenCV Mat to BufferedImage for display in the UI.
     */
    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = mat.channels() > 1 ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;
        byte[] buffer = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, buffer);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), buffer);
        return image;
    }
}
