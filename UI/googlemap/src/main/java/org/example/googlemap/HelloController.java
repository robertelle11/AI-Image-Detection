package org.example.googlemap;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ToggleButton;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.DragEvent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class HelloController {


@FXML
private ToggleButton toggleButton;

@FXML
private AnchorPane anchorPane1;

@FXML
private AnchorPane anchorPane2;

    @FXML
    private AnchorPane anchorPane3;


    @FXML
   private ImageView imageView;


    @FXML
    void dragimageover(DragEvent dragEvent){
        Dragboard dragboard=dragEvent.getDragboard();

        if(dragboard.hasFiles()||dragboard.hasImage()){

            dragEvent.acceptTransferModes(TransferMode.COPY);
        }
    }

    @FXML
    void dragimagedropped(DragEvent dragEvent) throws FileNotFoundException {
        Dragboard dragboard=dragEvent.getDragboard();
        if(dragboard.hasFiles()||dragboard.hasImage()){
imageView.setImage(new Image(new FileInputStream(dragboard.getFiles().getFirst())));
        }
    }


    @FXML
    protected void onclick(ActionEvent event) throws IOException {

        if (anchorPane1.isVisible()){
            anchorPane1.setVisible(false);
            anchorPane2.setVisible(false);
            anchorPane3.setVisible(false);
        }
        else{
            anchorPane1.setVisible(true);
            anchorPane2.setVisible(true);
            anchorPane3.setVisible(true);


        }


    }
}