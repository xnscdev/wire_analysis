package com.github.xnscdev.wa;

import io.scif.img.ImgSaver;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.ImgView;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.scijava.command.Command;
import org.scijava.io.location.FileLocation;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.File;

@Plugin(type = Command.class, menuPath = "Plugins>Wire Analysis")
public class WireAnalysis<T extends RealType<T>> implements Command {
    private final ImgSaver saver = new ImgSaver();

    @Parameter
    private Dataset currentData;

    @Parameter
    private UIService uiService;

    @Parameter
    private OpService opService;

    @Parameter(label = "Select output directory", style = "directory")
    private File outputDir;

    @Override
    public void run() {
        @SuppressWarnings("unchecked")
        ImgPlus<T> image = (ImgPlus<T>) currentData.getImgPlus();
        medianThreshold(image);
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
        String name = image.getName().replace(".tif", "_median.tif");
        FileLocation loc = new FileLocation(new File(outputDir, name));
        saver.saveImg(loc, converted);
        uiService.show(name, converted);
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
