/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree;

import classifier.id3.MyID3;
import classifier.MyJ48;
import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.ID3;
import weka.classifiers.Classifier;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Sekar Anglila/13514069 Diastuti Utami/13514071 Bervianto Leo
 * P/13514047
 */
public class Weka_decisionTree {

    private Instances data;
    private Instances filteredData;
    private Instances headerData;
    private Classifier classifier;
    private Classifier model;
    private final Scanner scan;

    public Weka_decisionTree() {
        scan = new Scanner(System.in);
    }

    public boolean isHaveData() {
        return data != null;
    }

    //Reader
    public void readData(String filename) {
        try {
            data = DataSource.read(filename);
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            headerData = new Instances(this.data, 0);
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("File tidak dapat dibuka.");
        }
    }

    public void readModel(String filename) {
        try {
            Object o[] = SerializationHelper.readAll(filename);
            this.model = (Classifier) o[0];
            this.headerData = (Instances) o[1];
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("File tidak dapat dibuka.");
        }
    }

    public void saveModel(String filename) {
        //Kamus Lokal
        if (this.model != null) {
            try {
                SerializationHelper.write(filename, new Object[]{this.model, this.headerData});
            } catch (Exception ex) {
                Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            System.out.println("Belum ada model. Tidak ada yang disimpan.");
        }
    }

    //Resample and Remove Attributes
    public void filterData() {
        try {
            //Kamus Lokal
            Resample filter = new Resample();

            //Algoritma
            filter.setInputFormat(this.data);
            this.filteredData = Filter.useFilter(this.data, filter);

        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void discretize() {
        try {
            Discretize discret = new Discretize();

            //Algoritma
            discret.setInputFormat(this.data);
            this.filteredData = Filter.useFilter(this.data, discret);
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void skemaTenFolds() {
        try {
            //Kamus Lokal
            System.out.print("Jumlah Fold: ");

            long seed = System.currentTimeMillis();
            int folds = scan.nextInt();
            int numInstances = this.filteredData.numInstances();
            if (folds > numInstances) {
                folds = numInstances;
                System.out.println("Folds using maximum instance");
            } else if (folds < 2) {
                folds = 2;
                System.out.println("Folds using minimum folds.");
            }
            Random rand = new Random(seed);
            Evaluation eval = new Evaluation(this.filteredData);
            this.classifier.buildClassifier(this.filteredData);
            this.model = this.classifier;
            eval.crossValidateModel(this.classifier, this.filteredData, folds, rand);
            //Menampilkan di Layar
            System.out.println();
            System.out.println(eval.toSummaryString("=== 10-fold-Cross-Validation ===", false));
            System.out.println(eval.toMatrixString());
            System.out.println();
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    public void skemaFullTraining() {
        try {
            //Kamus Lokal
            Evaluation eval = new Evaluation(this.filteredData);
            //Algoritma
            this.classifier.buildClassifier(filteredData);
            this.model = this.classifier;
            eval.evaluateModel(this.classifier, filteredData);
            //Menampilkan di Layar
            System.out.println();
            System.out.println("====================Results===================");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
            System.out.println();
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void splitTest() {
        try {
            System.out.print("Masukan berapa persen yang akan digunakan: ");
            int percent = scan.nextInt();
            if (percent > 0 && percent < 100) {
                int trainSize = (int) Math.round(this.filteredData.numInstances() * percent / 100.0);
                int testSize = this.data.numInstances() - trainSize;
                Instances train = new Instances(this.filteredData, 0, trainSize);
                Instances test = new Instances(this.filteredData, trainSize, testSize);
                Evaluation eval = new Evaluation(train);
                //Algoritma
                this.classifier.buildClassifier(train);
                this.model = this.classifier;
                eval.evaluateModel(this.classifier, test);
                //Menampilkan di Layar
                System.out.println();
                System.out.println("====================Results===================");
                System.out.println(eval.toSummaryString());
                System.out.println(eval.toMatrixString());
                System.out.println();
            } else {
                System.out.println("Persen harus diantara 0-100");
            }
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void skemaTestSet() {
        try {
            System.out.print("Masukan file test: ");
            String filename = scan.nextLine();
            Instances dataTest = DataSource.read(filename);
            if (dataTest.classIndex() == -1) {
                dataTest.setClassIndex(dataTest.numAttributes() - 1);
            }
            //Kamus Lokal
            Evaluation eval = new Evaluation(this.filteredData);
            //Algoritma
            this.classifier.buildClassifier(this.filteredData);
            this.model = this.classifier;
            eval.evaluateModel(this.classifier, dataTest);
            //Menampilkan di Layar
            System.out.println();
            System.out.println("====================Results===================");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
            System.out.println();
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("File tidak dapat dibuka.");
        }
    }

    public String classify(Instance newData) {
        try {
            double result = this.model.classifyInstance(newData);
            String result_string = newData.classAttribute().value((int) result);

            return result_string;
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }
        return "";
    }

    public void evaluatesModel() {
        try {
            //Kamus Lokal
            Evaluation eval = new Evaluation(this.data);

            //Algoritma
            eval.crossValidateModel(this.model, this.filteredData, 10, new Random(1));
            System.out.println(eval.toSummaryString("======================Results======================\n", true));
            System.out.println(eval.fMeasure(1) + " " + eval.recall(1));
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void classifyInstance() {
        try {
            System.out.print("Masukan file yang akan diklasifikasi: ");
            String filename = scan.nextLine();
            Instances dataTest = DataSource.read(filename);
            if (dataTest.classIndex() == -1) {
                dataTest.setClassIndex(dataTest.numAttributes() - 1);
            }
            //Kamus Lokal
            Evaluation eval = new Evaluation(this.filteredData);
            //Algoritma
            this.model.buildClassifier(this.filteredData);
            eval.evaluateModel(this.model, dataTest);
            //Menampilkan di Layar
            System.out.println();
            System.out.println("====================Results===================");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
            System.out.println();
        } catch (Exception ex) {
            Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    private void removeAttributes() {
        int i = 0;
        Enumeration<Attribute> attrb = this.filteredData.enumerateAttributes();
        while (attrb.hasMoreElements()) {
            System.out.format("%d. %s\n", i, attrb.nextElement().name());
            i++;
        }
        System.out.print("Attribute yang ingin anda hapus? ");
        int removeIndex = scan.nextInt();
        if (removeIndex >= 0 && removeIndex < i) {
            try {
                Remove rm = new Remove();
                rm.setAttributeIndices(String.valueOf(removeIndex));
                rm.setInputFormat(this.filteredData);
                this.filteredData = Filter.useFilter(this.filteredData, rm);
            } catch (Exception ex) {
                Logger.getLogger(Weka_decisionTree.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            System.out.println("Tidak dapat menghapus.");
        }
    }

    private void showHeaderFile() {
        //Menampilkan ke layar
        System.out.println();
        System.out.print("Header File: \n");
        System.out.println(this.headerData);
        System.out.println();
    }

    public int Option() {
        //Kamus Lokal
        int pil1;
        Scanner input = new Scanner(System.in);

        //Algoritma
        System.out.println();
        System.out.print("Tentukan apa yang ingin dilakukan: \n");
        System.out.print("1. Skema 10-fold cross validation \n");
        System.out.print("2. Skema Full Training \n");
        System.out.print("3. Menguji dengan set test \n");
        System.out.print("4. Percentage split\n");
        System.out.print("5. Load Model \n");
        System.out.print("6. Classify Instance \n");
        System.out.print("7. Remove Attributes \n");
        System.out.print("8. Exit \n");
        System.out.print("Pilihan anda: ");
        pil1 = input.nextInt();
        return pil1;
    }

    public void chooseClassifier() {
        int pilihan;
        Scanner input = new Scanner(System.in);

        //Algoritma
        System.out.println();
        System.out.print("Pilih Classifier: \n");
        System.out.print("1. myID3 \n");
        System.out.print("2. myJ48 \n");
        System.out.print("3. J48 - Weka \n");
        System.out.print("4. ID3 - Weka \n");
        System.out.print("Pilihan anda (default ID3 Weka, another 1-4 will choose ID3 Weka) : ");
        pilihan = input.nextInt();

        switch (pilihan) {
            case 1:
                this.classifier = new MyID3();
                filterData();
                break;
            case 2:
                this.classifier = new MyJ48();
                filterData();
                break;
            case 3:
                this.classifier = new J48();
                filterData();
                break;
            default:
                this.classifier = new ID3();
                discretize();
                break;
        }
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        //Kamus
        Weka_decisionTree TW = new Weka_decisionTree();
        Scanner input = new Scanner(System.in);

        //Algoritma
        System.out.println("-------------------------------------------------------------------\n");
        System.out.println("                          WEKA TUBES 01                            \n");
        System.out.println("-------------------------------------------------------------------\n");
        System.out.println();

        //Membaca file
        System.out.println();
        System.out.println("Membaca File");
        System.out.print("Masukkan nama file: ");
        while (!TW.isHaveData()) {
            String filename = input.nextLine();
            TW.readData(filename);
        }
        TW.showHeaderFile();

        //Discretize
        System.out.println("===============Resample==============");

        TW.showHeaderFile();

        TW.chooseClassifier();

        //Pilihan Pengelolaan data
        int pil = TW.Option();

        //Loop pilihan 2
        while (pil != 6) {
            switch (pil) {
                case 1:
                    //Skema CV 10 Folds
                    TW.showHeaderFile();
                    TW.skemaTenFolds();
                    TW.saveModel("cv-model.model");
                    break;
                case 2:
                    //Skema Full Training
                    //Menampilkan ke layar
                    TW.showHeaderFile();
                    //Skema Full Training
                    TW.skemaFullTraining();
                    TW.saveModel("full-model.model");
                    //Pilihan
                    break;
                case 3:
                    // Test Set
                    TW.showHeaderFile();
                    TW.skemaTestSet();
                    TW.saveModel("test-set-model.model");
                    break;
                case 4:
                    // Split Test
                    TW.showHeaderFile();
                    TW.splitTest();
                    TW.saveModel("split.model");
                    break;
                case 5:
                    //Load File
                    //Load Model
                    System.out.print("Masukkan nama File: ");
                    String filename2 = input.nextLine();
                    TW.readModel(filename2);
                    System.out.print("Berhasil dibaca!\n\n");
                    TW.evaluatesModel();
                    System.out.println();
                    break;
                case 6:
                    //Clasify one unseen data
                    TW.classifyInstance();
                    break;
                case 7:
                    // Remove Attribute
                    TW.removeAttributes();
                    break;
                default:
                    //Error Message
                    System.out.println("Pilihan tidak valid!\n");
                    break;
            }
            TW.chooseClassifier();
            pil = TW.Option();
        }
        //Exit
        System.out.println();
        System.out.println("Terima Kasih!\n");
    }
}
