package com.empty.djl;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.apache.commons.compress.utils.IOUtils;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@RestController
public class ImageDetectController {

    @PostMapping(value = "/upload", produces = MediaType.TEXT_HTML_VALUE)
    public ResponseEntity diagnose(@RequestParam("file") MultipartFile file) throws ModelException, TranslateException, IOException {
        byte[] bytes = file.getBytes();
        Path imageFile = Paths.get(file.getOriginalFilename());
        Files.write(imageFile, bytes);
        return predict(imageFile);
    }


    public ResponseEntity predict(Path imageFile) throws IOException, ModelException, TranslateException {
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        String backbone;
        if ("TensorFlow".equals(Engine.getDefaultEngineName())) {
            backbone = "mobilenet_v2";
        } else {
            backbone = "resnet50";
        }

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("backbone", backbone)
                        .optEngine(Engine.getDefaultEngineName())
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                return saveBoundingBoxImage(img, detection);
            }
        }
    }


    private ResponseEntity saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("src/main/resources");
        Files.createDirectories(outputDir);

        //Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        try {
            Image newImage = img.duplicate();
            newImage.drawBoundingBoxes(detection);

            Path imagePath = outputDir.resolve("detected.png");
            newImage.save(Files.newOutputStream(imagePath), "png");
        }
        catch (Exception e) {
            System.out.println("Exception occurred while saving image" + e.getMessage());
            e.printStackTrace();
        }
        String fileDownloadUri = ServletUriComponentsBuilder.fromCurrentContextPath()
                .path("get")
                .toUriString();
        return ResponseEntity.ok(fileDownloadUri);
    }


    @GetMapping(
            value = "/get",
            produces = MediaType.IMAGE_PNG_VALUE
    )
    public @ResponseBody
    byte[] getImageWithMediaType() throws IOException {
        InputStream in = new ClassPathResource(
                "detected.png").getInputStream();
        return IOUtils.toByteArray(in);
    }

}