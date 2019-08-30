import ai.skymind.ApiClient;
import ai.skymind.ApiException;
import ai.skymind.Configuration;
import ai.skymind.auth.ApiKeyAuth;
import ai.skymind.skil.DefaultApi;
import com.google.common.base.Stopwatch;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class BenchmarkSKILResNet {
    public static void main(String... args) throws Exception {
        String host = "http://localhost:9008";
        String apiKey = null;
        String modelName = "resnet50";
        String deploymentName = "test";
        String numThreads = "8";
        String examples = "1000";
        String imagePath = null;

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
            } else if ("--image".equals(arg) || "-i".equals(arg)) {
                imagePath = value;
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
        if (imagePath == null) {
            System.out.println("Argument --image is required.");
            usage();
            System.exit(1);
        }

        ApiClient defaultClient = Configuration.getDefaultApiClient();

        defaultClient.setBasePath(host);

        ApiKeyAuth x_api_key = (ApiKeyAuth) defaultClient.getAuthentication("x_api_key");
        x_api_key.setApiKey(apiKey);

        DefaultApi apiInstance = new DefaultApi(defaultClient);

        ExecutorService threads = Executors.newFixedThreadPool(Integer.parseInt(numThreads));
        Stopwatch wholeStopwatch = Stopwatch.createStarted();

        ArrayList<Future<?>> jobs = new ArrayList<>();

        // Variables shared across threads.
        final List<Long> timings = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger total = new AtomicInteger(0);
        final String deployment = deploymentName;
        final String model = modelName;
        final File image = new File(imagePath);
        final int numExamples = Integer.parseInt(examples);

        for (int i = 0; i < numExamples; i++) {
            jobs.add(threads.submit(() -> {
                try {
                    Stopwatch stopwatch = Stopwatch.createStarted();

                    apiInstance.classifyimage(deployment, "default", model, image);
                    long elapsedRequest = stopwatch.elapsed(TimeUnit.MILLISECONDS);

                    timings.add(elapsedRequest);

                    total.incrementAndGet();
                } catch (ApiException e) {
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
        System.out.println("Took " + elapsed + " seconds. (" + (float) total.get() / elapsed + " rps)");
    }

    private static void usage() {
        System.out.println("Usage: java -cp skil-benchmark-bin.jar BenchmarkSKILResNet [args]");
        System.out.println("    --host | -h         Hostname of SKIL server (default: http://localhost:9008)");
        System.out.println("    --key | -k          API Key to use for requests. [Required] ");
        System.out.println("    --model | -m        Name of model server to send request to (default: mnist)");
        System.out.println("    --deployment | -d   Name of deployment to send request to (default: test)");
        System.out.println("    --threads | -t      Number of threads to use (default: 8)");
        System.out.println("    --examples | -e     Number of requests to send (default: 1000)");
        System.out.println("    --image | -i        Path to image file to use [Required]");
    }

    private static long percentile(Double p, List<Long> seq) {
        int k = (int) Math.ceil((seq.size() - 1) * (p / 100.0));

        return seq.get(k);
    }
}
