/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree;

import java.util.Random;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Scanner;
import weka.classifiers.Classifier;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
/**
 *
 * @author Sekar Anglila/13514069
 *          Diastuti Utami/13514071
 *          Bervianto Leo P/13514047
 */
public class Weka_decisionTree {

    //Reader
    public Instances readFile(String filename) throws Exception {
        //Kamus Lokal
        Instances data = null;
        BufferedReader reader;

        //Algoritma
        reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
        data.setClassIndex(data.numAttributes() - 1);
        reader.close();

        return data;
    }

    //Discretize
    public Instances filterData(Instances data) throws Exception {
        //Kamus Lokal
        Resample filter = new Resample();
        Instances filterRes;
        Remove r = new Remove();

        //Algoritma
        //r.setAttributeIndices("28");
        r.setInputFormat(data);
        filterRes = Filter.useFilter(data, r);

        return filterRes;
    }

    public Classifier skemaTenFolds(Instances data) throws Exception {
        //Kamus Lokal
        J48 dt = new J48();
        int seed = 1;
        int folds = 10;
        Random rand = new Random(seed);
        Instances randData = new Instances(data);

        //Algoritma
        randData.randomize(rand);
        //Stratify
        if (randData.classAttribute().isNominal()){
            randData.stratify(folds);
        }
        Evaluation eval = new Evaluation(randData);
        for(int n = 0; n < folds; n++){
            Instances train = randData.trainCV(folds, n);
            Instances test = randData.testCV(folds, n);
            dt.buildClassifier(train);
            eval.evaluateModel(dt, test);
        }
        //Menampilkan di Layar
        System.out.println();
        System.out.println(eval.toSummaryString("=== 10-fold-Cross-Validation ===", false));
        System.out.println(eval.toMatrixString());
        System.out.println();

        return dt;
    }

    public Classifier skemaFullTraining(Instances data) throws Exception {
        //Kamus Lokal
        J48 dt = new J48();
        Evaluation eval = new Evaluation(data);

        //Algoritma
        dt.buildClassifier(data);
        eval.evaluateModel(dt, data);
        //Menampilkan di Layar

        System.out.println();
        System.out.println("====================Results===================");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
        System.out.println();

        return dt;
    }

    public Classifier readObject(String filename) throws Exception {
        //Kamus Lokal
        Classifier ClassRes;
        ObjectInputStream reader = new ObjectInputStream(new FileInputStream(filename));

        //Algoritma
        ClassRes = (Classifier) reader.readObject();
        reader.close();
        return ClassRes;
    }

    public void saveObject(Classifier C, String filename) throws Exception {
        //Kamus Lokal
        ObjectOutputStream writer = new ObjectOutputStream(new FileOutputStream(filename));

        //Algoritma
        writer.writeObject(C);
        writer.flush();
        writer.close();
    }

    public String classify(Instance data, Classifier C) throws Exception{
        double result = C.classifyInstance(data);
        String result_string = data.classAttribute().value((int) result);

        return result_string;
    }

    public void Evaluates(Instances dataDisc, Classifier classRes) throws Exception{
        //Kamus Lokal
        Evaluation eval = new Evaluation(dataDisc);

        //Algoritma
        eval.crossValidateModel(classRes, dataDisc, 10, new Random(1));
        System.out.println(eval.toSummaryString("======================Results======================\n",true));
        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
    }

    public int Option(){
        //Kamus Lokal
        int pil1;
        Scanner input = new Scanner(System.in);

        //Algoritma
        System.out.println();
        System.out.print("Tentukan apa yang ingin dilakukan: \n");
        System.out.print("1. Skema 10-fold cross validation \n");
        System.out.print("2. Skema Full Training \n");
        System.out.print("3. Load Model \n");
        System.out.print("4. Input Instance \n");
        System.out.print("5. Exit \n");
        System.out.print("Pilihan anda: ");
        pil1 = input.nextInt();
        return pil1;
    }

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception{
        //Kamus
        Weka_decisionTree TW = new Weka_decisionTree();
        Scanner input = new Scanner(System.in);
        String filename = null;
        String filename2 = null;
        int pil = 0;
        Instances data, dataDisc;
        Classifier classRes = null;

        //Algoritma
        System.out.println("-------------------------------------------------------------------\n");
        System.out.println("                          WEKA TUBES 01                            \n");
        System.out.println("-------------------------------------------------------------------\n");
        System.out.println();

        //Membaca file
        System.out.println();
        System.out.println("Membaca File");
        System.out.print("Masukkan nama file: ");
        filename = input.nextLine();
        data = TW.readFile(filename);

        //Menampilkan ke layar
        System.out.println();
        System.out.print("Header File: \n");
        System.out.println(new Instances (data, 0));
        System.out.println();

        //Discretize
        System.out.println("===============Resample==============");
        dataDisc = TW.filterData(data);
        System.out.println(new Instances (dataDisc, 0));

        //Pilihan Pengelolaan data
        pil = TW.Option();

        //Loop pilihan 2
        while (pil != 5){
            if (pil == 1){ //Skema Ten Folds
                //Menampilkan ke layar
                System.out.println();
                System.out.print("Header File: \n");
                System.out.println(new Instances (dataDisc, 0));
                System.out.println();

                //Skema 10 Folds
                classRes = TW.skemaTenFolds(dataDisc);
                System.out.println();

                //Pilihan
                pil = TW.Option();
            } else if (pil == 2){ //Skema Full Training
                //Menampilkan ke layar
                System.out.println();
                System.out.print("Header File: \n");
                System.out.println(new Instances (dataDisc, 0));
                System.out.println();

                //Skema Full Training
                classRes = TW.skemaFullTraining(dataDisc);
                System.out.println();

                //Menyimpan Model
                System.out.println("===============Menyimpan Model==============");
                System.out.print("File name: ");
                filename = input.next();
                TW.saveObject(classRes, filename);
                System.out.print("Berhasil disimpan!\n");

                //Pilihan
                pil = TW.Option();
            } else if (pil == 3) { //Load File
                //Load Model
                System.out.print("Masukkan nama File: ");
                filename2 = input.next();
                Classifier cls = TW.readObject(filename2);
                System.out.print("Berhasil dibaca!\n\n");

                //Evaluate
                TW.Evaluates(dataDisc, classRes);

                //Pilihan
                System.out.println();
                pil = TW.Option();
            } else if (pil == 4){ //Add Instance
                //input instance
                System.out.println();
                classRes.buildClassifier(data);
                Instance insert = new DenseInstance(4);
                insert.setDataset(data);
                for(int i=0; i<data.classIndex(); i++){
                    System.out.print("Atribut ke-" + i+1+"= ");
                    float atribut = input.nextFloat();
                    insert.setValue(i,atribut);
                }

                //classify
                String baru = TW.classify(insert, classRes);
                System.out.println("Hasil Klasifikasi: " + baru);

                //Pilihan
                System.out.println();
                pil = TW.Option();
            } else {
                //Error Message
                System.out.println("Pilihan tidak valid!\n");

                //Pilihan Pengelolaan data
                pil = TW.Option();
            }
        }
        //Exit
        System.out.println();
        System.out.println("Terima Kasih!\n");
    }
    
}
