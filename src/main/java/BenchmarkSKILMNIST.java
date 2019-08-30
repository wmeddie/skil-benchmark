import ai.skymind.ApiClient;
import ai.skymind.ApiException;
import ai.skymind.Configuration;
import ai.skymind.auth.ApiKeyAuth;
import ai.skymind.skil.DefaultApi;
import ai.skymind.skil.model.ClassificationResult;
import ai.skymind.skil.model.INDArray;
import ai.skymind.skil.model.Prediction;
import com.google.common.base.Stopwatch;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class BenchmarkSKILMNIST {
    public static void main(String... args) throws Exception {
        String host = "http://localhost:9008";
        String apiKey = null;
        String modelName = "mnist";
        String deploymentName = "test";
        String numThreads = "8";
        String examples = "1000";

        List<String> argList = new ArrayList<>(args.length);
        argList.addAll(Arrays.asList(args));
        while (argList.size() >= 2) {
            String arg = argList.get(0);
            String value = argList.get(1);

            if ("--host".equals(arg) || "-h".equals(arg)) {
                host = value;
            } else if ("--key".equals(arg) || "-k".equals(arg)) {
                apiKey = value;
            } else if ("--model".equals(arg) || "-m".equals(arg)) {
                modelName = value;
            } else if ("--deployment".equals(arg) || "-d".equals(arg)) {
                deploymentName = value;
            } else if ("--threads".equals(arg) || "-t".equals(arg)) {
                numThreads = value;
            } else if ("--examples".equals(arg) || "-e".equals(arg)) {
                examples = value;
            } else {
                System.out.println("Invalid argument: " + arg);
                usage();
                System.exit(1);
            }

            argList.remove(0);
            argList.remove(0);
        }
        if (apiKey == null) {
            System.out.println("Argument --key is required.");
            usage();
            System.exit(1);
        }

        ApiClient defaultClient = Configuration.getDefaultApiClient();

        defaultClient.setBasePath(host);

        ApiKeyAuth x_api_key = (ApiKeyAuth) defaultClient.getAuthentication("x_api_key");
        x_api_key.setApiKey(apiKey);

        DefaultApi apiInstance = new DefaultApi(defaultClient);

        MnistDataSetIterator data = new MnistDataSetIterator(1, Integer.parseInt(examples));

        ExecutorService threads = Executors.newFixedThreadPool(Integer.parseInt(numThreads));
        Stopwatch wholeStopwatch = Stopwatch.createStarted();

        ArrayList<Future<?>> jobs = new ArrayList<>();

        // Variables shared across threads.
        final List<Long> timings = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger total = new AtomicInteger(0);
        final AtomicInteger right = new AtomicInteger(0);
        final String deployment = deploymentName;
        final String model = modelName;

        while (data.hasNext()) {
            final DataSet ds = data.next();

            jobs.add(threads.submit(() -> {
                try {
                    Prediction prediction = new Prediction();
                    INDArray array = new INDArray();

                    array.array(Nd4jBase64.base64String(ds.getFeatures()));
                    prediction.setPrediction(array);
                    prediction.setId(UUID.randomUUID().toString());

                    Stopwatch stopwatch = Stopwatch.createStarted();

                    ClassificationResult res = apiInstance.classify(deployment, "default", model, prediction);
                    long elapsedRequest = stopwatch.elapsed(TimeUnit.MILLISECONDS);

                    timings.add(elapsedRequest);

                    int truth = Nd4j.argMax(ds.getLabels(), -1).max(-1).data().getInt(0);
                    int pred = res.getResults().get(0);

                    total.incrementAndGet();
                    if (truth == pred) {
                        right.incrementAndGet();
                    }

                } catch (ApiException | IOException e) {
                    throw new RuntimeException(e);
                }
            }));
        }

        try {
            System.out.println("Waiting for threads");
            int i = 0;
            for (Future<?> job : jobs) {
                job.get();
                System.out.print("\rCompleted " + (i + 1) + " of " + examples);
                i++;
            }
            System.out.println("\nFinished.");
        } finally {
            threads.shutdown();
        }

        timings.sort(Long::compareTo);

        long elapsed = wholeStopwatch.elapsed(TimeUnit.SECONDS);
        float acc = (float) right.get() / total.get();


        double inferenceMean = timings.stream().reduce(0L, Long::sum) / (double)timings.size();
        long inferenceMin = timings.get(0);
        long inferenceMax = timings.get(timings.size() - 1);
        long oneNine = percentile(90.0, timings);
        long twoNines = percentile(99.0, timings);
        long threeNines = percentile(99.9, timings);
        long fourNines = percentile(99.99, timings);
        long fiveNines = percentile(99.999, timings);
        long sixNines = percentile(99.9999, timings);

        System.out.println("Mean " + inferenceMean + " ms");
        System.out.println("Median " + timings.get(timings.size() / 2) + " ms");
        System.out.println("Min " + inferenceMin + " ms");
        System.out.println("Max " + inferenceMax + " ms");
        System.out.println("90 Percentile " + oneNine + " ms");
        System.out.println("99 Percentile " + twoNines + " ms");
        System.out.println("99.9 Percentile " + threeNines + " ms");
        System.out.println("99.99 Percentile " + fourNines + " ms");
        System.out.println("99.999 Percentile " + fiveNines + " ms");
        System.out.println("99.9999 Percentile " + sixNines + " ms");

        System.out.println();
        System.out.printf("Accuraccy %.4f", acc);
        System.out.println();
        System.out.println("Took " + elapsed + " seconds. (" + (float) total.get() / elapsed + " rps)");
    }

    private static void usage() {
        System.out.println("Usage: java -cp skil-benchmark-bin.jar BenchmarkSKILMNIST [args]");
        System.out.println("    --host | -h         Hostname of SKIL server (default: http://localhost:9008)");
        System.out.println("    --key | -k          API Key to use for requests. [Required] ");
        System.out.println("    --model | -m        Name of model server to send request to (default: mnist)");
        System.out.println("    --deployment | -d   Name of deployment to send request to (default: test)");
        System.out.println("    --threads | -t      Number of threads to use (default: 8)");
        System.out.println("    --examples | -e     Number of requests to send (default: 1000)");
    }

    private static long percentile(Double p, List<Long> seq) {
        int k = (int) Math.ceil((seq.size() - 1) * (p / 100.0));

        return seq.get(k);
    }
}
