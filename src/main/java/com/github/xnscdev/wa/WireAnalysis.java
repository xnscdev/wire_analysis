package com.github.xnscdev.wa;

import io.scif.config.SCIFIOConfig;
import io.scif.img.ImgSaver;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imglib2.IterableInterval;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.scijava.command.Command;
import org.scijava.io.location.FileLocation;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.*;
import java.util.stream.Stream;

@Plugin(type = Command.class, menuPath = "Plugins>Wire Analysis")
public class WireAnalysis<T extends RealType<T>> implements Command {
    private final ImgSaver saver = new ImgSaver();
    private final SCIFIOConfig config = new SCIFIOConfig();

    public WireAnalysis() {
        config.writerSetFailIfOverwriting(false);
    }

    @Parameter
    private Dataset currentData;

    @Parameter
    private UIService uiService;

    @Parameter
    private OpService opService;

    @Parameter(label = "Select output directory", style = "directory")
    private File outputDir;

    @Parameter(label = "Python interpreter path")
    private File pythonPath;

    @Parameter(label = "Pixels per micrometer")
    private int pixelsPerMicro;

    @Override
    public void run() {
        @SuppressWarnings("unchecked")
        ImgPlus<T> image = (ImgPlus<T>) currentData.getImgPlus();
        medianThreshold(image);
        runPython("small_features", getProcessedImagePath(image, "median"), String.valueOf(pixelsPerMicro), outputDir.getAbsolutePath());
        segmentation(image);
    }

    private String getProcessedImageName(ImgPlus<T> image, String suffix) {
        return image.getName().replace(".tif", "_" + suffix + ".tif");
    }

    private String getProcessedImagePath(ImgPlus<T> image, String suffix) {
        return outputDir.getAbsolutePath() + File.separator + getProcessedImageName(image, suffix);
    }

    private void medianThreshold(ImgPlus<T> image) {
        ImgFactory<BitType> bitFactory = image.factory().imgFactory(new BitType());
        Img<BitType> median = bitFactory.create(image);
        opService.threshold().localMedianThreshold(median, image, new RectangleShape(15, true), 0);
        Img<BitType> inverted = opService.create().img(median);
        opService.image().invert(inverted, median);
        Img<UnsignedByteType> converted = opService.convert().uint8(inverted);
        for (UnsignedByteType pixel : converted) {
            pixel.mul(255);
        }
        String name = getProcessedImageName(image, "median");
        saveImage(name, converted);
        uiService.show(name, converted);
    }

    private void segmentation(ImgPlus<T> image) {
        IterableInterval<BitType> segmented = opService.threshold().minimum(image);
        Img<UnsignedByteType> converted = opService.convert().uint8(segmented);
        for (UnsignedByteType pixel : converted) {
            pixel.mul(255);
        }
        String name = getProcessedImageName(image, "seg");
        saveImage(name, converted);
    }

    private void saveImage(String name, Img<? extends RealType<?>> converted) {
        FileLocation loc = new FileLocation(new File(outputDir, name));
        saver.saveImg(loc, converted, config);
    }

    private void runPython(String name, String... args) {
        try (InputStream inputStream = this.getClass().getResourceAsStream("/" + name + ".py")) {
            Runtime rt = Runtime.getRuntime();
            String[] prefix = {pythonPath.getAbsolutePath(), "-"};
            String[] command = Stream.of(prefix, args).flatMap(Stream::of).toArray(String[]::new);
            Process process = rt.exec(command);
            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) != -1) {
                process.getOutputStream().write(buffer, 0, length);
            }
            process.getOutputStream().close();
            int status = process.waitFor();
            StringBuilder errorString = new StringBuilder();
            String line;
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            while ((line = errorReader.readLine()) != null) {
                errorString.append(line).append('\n');
            }
            errorReader.close();
            if (status != 0)
                throw new RuntimeException("Python process exited with nonzero status code:\n" + errorString);
        }
        catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) throws Exception {
        ImageJ ij = new ImageJ();
        ij.ui().showUI();
        File file = ij.ui().chooseFile(null, "open");
        if (file != null) {
            Dataset dataset = ij.scifio().datasetIO().open(file.getPath());
            ij.ui().show(dataset);
            ij.command().run(WireAnalysis.class, true);
        }
    }
}
