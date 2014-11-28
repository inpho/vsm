package org.knowceans.gibbstest;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class FileArrayProvider {

    public static int[][] readFile(String filename) throws IOException {
        /*
        Reads file that contains corpus.view_contexts(ctx_type),
        list of arrays. This returns the int[][] for LdaGibbsSampler
        documents.
        */
        FileReader fileReader = new FileReader(filename);
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        
        // List<String> lines = new ArrayList<String>();
        List<List<Integer>> lines = new ArrayList<List<Integer>>();
        String line = null;
        
        List<Integer> a = new ArrayList<Integer>();
        while ((line = bufferedReader.readLine()) != null) {
            if (line.length() > 0) { // String
                int item = Integer.parseInt(line);
                a.add(item);
            } else {
                lines.add(a);
                a = new ArrayList<Integer>();
            }
        }
        bufferedReader.close();
        
        int[][] arr = new int[lines.size()][];
        for (int i=0; i<lines.size(); i++) {
            
            List<Integer> subli = lines.get(i);
            int[] blankarr = new int[subli.size()];
            for (int j=0; j<subli.size(); j++) {
                blankarr[j] = subli.get(j).intValue();
            }
            arr[i] = blankarr;
        }
        
        return arr;
    }
       
    public static void writeDoubleFile(double[][] data, String filename) throws IOException {
         /*
        Need a function to write a .txt containing Z, phi, and theta
        for python to use. General function to be used for Z, phi, theta.
        @param filename : file to write to
        */
        
        FileWriter fileWriter = new FileWriter(filename);
        for (double[] d : data) {
            for (double i : d) {
                String s = String.valueOf(i);
                fileWriter.write(s + ",");
            }
            fileWriter.write('\n');
        }
        fileWriter.close();
    }


    public static void writeIntFile(int[][] data, String filename) throws IOException {
         /*
        @param filename : file to write to
        */
        
        FileWriter fileWriter = new FileWriter(filename);
        for (int[] d : data) {
            for (int i : d) {
                String s = String.valueOf(i);
                fileWriter.write(s + ",");
            }
            fileWriter.write('\n');
        }
        fileWriter.close();
    }

    public static void writeStrFile(String data, String filename) throws IOException {
        
        FileWriter fileWriter = new FileWriter(filename);
        fileWriter.write(data);
        fileWriter.close();
    }


    public static void main(String[] args) throws IOException {
        // test readFile. 
        int[][] arr = readFile("testcorp.txt");
        System.out.println(arr);
    }
}
