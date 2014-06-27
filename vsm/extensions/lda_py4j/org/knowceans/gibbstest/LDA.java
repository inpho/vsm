package org.knowceans.gibbstest;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.io.IOException;
import java.util.*;
import java.io.StringWriter;

import py4j.GatewayServer;


public class LDA {

    private static FileArrayProvider fap;
    private static LdaGibbsSampler lda;

    public LDA(String corpFile) throws IOException {
        
        FileArrayProvider fap = new FileArrayProvider();
        this.fap = fap;

        int[][] documents = fap.readFile(corpFile);
        
        List<Integer> vli = new ArrayList<Integer>();
        for (int[] d : documents) {
            for (int i : d) {
                if (!vli.contains(i)) {
                    vli.add(i);
                }
            }
        }

        int V = vli.size();
        int M = documents.length;
        System.out.println("V, M "+ V + " " + M);
        LdaGibbsSampler lda = new LdaGibbsSampler(documents, V);
        
        this.lda = lda;
    }
   
    public LdaGibbsSampler getLda() {
        return this.lda;
    }

    public FileArrayProvider getFap() {
        return this.fap;
    }
    
    public static void sample(int iter, int K, double alpha, double beta)  {
        // configure(iter, burnin, thinInterval, sampleLag) default values
        // from LdaGibbsSampler example.
        lda.configure(iter, 2000, 100, 10);
        lda.gibbs(K, alpha, beta); 
    }
    
    public static void main(String[] args) throws IOException {
        // Note: iter=1000   returns NaN for all phi, theta
        String corpfile = args[0];
        LDA ldai = new LDA(corpfile);
        GatewayServer gatewayServer = new GatewayServer(ldai);
        gatewayServer.start();
        System.out.println("Gateway Server Started!");
   }
    
    public void writeMeta(int iter, int K, double alpha, double beta, 
           String metaFile) throws IOException {
       
        String s = "";
        s += "K," + K + "\n";
        s += "iteration," + iter + "\n";
        s += "m_words," + this.getLda().V + "\n";
        s += "doc_prior," + alpha + "\n";
        s += "top_prior," + beta + "\n";
        // add dummy values
        s += "inv_top_sums," + "0.0\n";
        s += "log_probs," + "0.0\n";
        
        this.getFap().writeStrFile(s, metaFile);
    }
}
