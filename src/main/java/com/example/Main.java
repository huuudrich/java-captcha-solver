package com.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class Main {
    static {
        System.loadLibrary("opencv_java480");
    }

    public static void main(String[] args) {
        Mat mainImage = Imgcodecs.imread("mainBg.jpg");
        Mat puzzleImage = puzzleDetector("puzzle.jpg");

        Mat puzzleGray = new Mat();
        Imgproc.cvtColor(puzzleImage, puzzleGray, Imgproc.COLOR_BGR2GRAY);

        Mat edgesPuzzle = new Mat();
        Imgproc.Canny(puzzleGray, edgesPuzzle, 100, 200);

        Mat mainGray = new Mat();
        Imgproc.cvtColor(mainImage, mainGray, Imgproc.COLOR_BGR2GRAY);

        Mat edgesMain = new Mat();
        Imgproc.Canny(mainGray, edgesMain, 100, 200);

        Mat result = new Mat(edgesMain.rows() - edgesPuzzle.rows() + 1, edgesMain.cols() - edgesPuzzle.cols() + 1, CvType.CV_32FC1);

        Imgproc.matchTemplate(edgesMain, edgesPuzzle, result, Imgproc.TM_CCOEFF_NORMED);

        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
        Point matchLoc = mmr.maxLoc;

        Imgproc.rectangle(mainImage, matchLoc, new Point(matchLoc.x + puzzleImage.cols(), matchLoc.y + puzzleImage.rows()), new Scalar(0, 255, 0), 2);

        Imgcodecs.imwrite("result.png", mainImage);
    }


    public static Mat puzzleDetector(String path) {
        Mat src = Imgcodecs.imread(path, Imgcodecs.IMREAD_UNCHANGED);

        List<Mat> channels = new ArrayList<>(4);
        Core.split(src, channels);
        Mat alphaChannel = channels.get(3);

        int minX = alphaChannel.cols();
        int minY = alphaChannel.rows();
        int maxX = 0;
        int maxY = 0;
        for (int y = 0; y < alphaChannel.rows(); y++) {
            for (int x = 0; x < alphaChannel.cols(); x++) {
                double alpha = alphaChannel.get(y, x)[0];
                if (alpha > 0) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        Rect rect = new Rect(minX, minY, maxX - minX, maxY - minY);

        return src.submat(rect);
    }
}