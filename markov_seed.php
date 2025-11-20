#!/usr/bin/env php
<?php
declare(strict_types=1);

mb_internal_encoding('UTF-8');

class ModelStats
{
    public int $nGrams = 0;
    public int $totalTransitions = 0;
    public float $avgTransitions = 0.0;
    public int $maxTransitions = 0;
    public int $minTransitions = -1;
    public int $deadEnds = 0;

    public function toArray(): array
    {
        return [
            'nGrams' => $this->nGrams,
            'totalTransitions' => $this->totalTransitions,
            'avgTransitions' => $this->avgTransitions,
            'maxTransitions' => $this->maxTransitions,
            'minTransitions' => $this->minTransitions,
            'deadEnds' => $this->deadEnds,
        ];
    }
}

class MarkovSeedGenerator
{
    public int $n;
    public array $model;
    public string $text;
    public bool $verbose;
    public array $logMessages;
    public bool $useSecureRand;

    public function __construct(int $n = 3, bool $verbose = false, bool $useSecureRand = true)
    {
        if ($n <= 0) {
            throw new InvalidArgumentException('n must be positive');
        }
        $this->n = $n;
        $this->model = [];
        $this->text = '';
        $this->verbose = $verbose;
        $this->logMessages = [];
        $this->useSecureRand = $useSecureRand;
    }

    public function log(string $format, mixed ...$args): void
    {
        if ($this->verbose) {
            $message = vsprintf($format, $args);
            $timestamp = (new DateTimeImmutable())->format(DateTime::ATOM);
            $entry = sprintf('[%s] %s', $timestamp, $message);
            $this->logMessages[] = $entry;
            fwrite(STDERR, $entry . PHP_EOL);
        }
    }

    public function getLogs(): array
    {
        return $this->logMessages;
    }

    public function clearLogs(): void
    {
        $this->logMessages = [];
    }

    private function secureRandInt(int $n): int
    {
        if ($n <= 0) {
            return 0;
        }
        if ($this->useSecureRand) {
            try {
                return random_int(0, $n - 1);
            } catch (Throwable $e) {
                // Fall through to fallback
            }
        }
        // Better fallback without bias
        $bytes = random_bytes(8);
        $value = unpack('J', $bytes)[1];
        return abs($value) % $n;
    }

    private function sanitizeText(string $text): string
    {
        return preg_replace('/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/u', '', $text);
    }

    public function train(string $inputText): void
    {
        $text = $this->sanitizeText($inputText);
        $textLength = mb_strlen($text);
        
        if ($textLength <= $this->n) {
            throw new InvalidArgumentException(sprintf('text length %d must be greater than n %d', $textLength, $this->n));
        }
        
        $this->text = $text;
        $this->model = [];
        
        for ($i = 0; $i <= $textLength - $this->n - 1; $i++) {
            $key = mb_substr($text, $i, $this->n);
            $nextChar = mb_substr($text, $i + $this->n, 1);
            
            if (!isset($this->model[$key])) {
                $this->model[$key] = [];
            }
            $this->model[$key][] = $nextChar;
        }
        
        $this->log('Trained model with %d n-grams', count($this->model));
    }

    public function trainFromFile(string $filename): void
    {
        if (!is_readable($filename)) {
            throw new RuntimeException(sprintf('failed to open training file: %s', $filename));
        }
        
        $size = filesize($filename);
        if ($size === false) {
            throw new RuntimeException(sprintf('failed to get file size: %s', $filename));
        }
        
        if ($size === 0) {
            throw new RuntimeException('training file is empty');
        }
        
        $maxSize = 100 * 1024 * 1024;
        if ($size > $maxSize) {
            throw new RuntimeException(sprintf('file too large: %d bytes', $size));
        }
        
        $this->log('Training from file: %s', $filename);
        
        $content = file_get_contents($filename);
        if ($content === false) {
            throw new RuntimeException(sprintf('failed to read training file: %s', $filename));
        }
        
        $this->train($content);
    }

    public function generate(int $length, ?string $startWith = null): string
    {
        if (count($this->model) === 0) {
            throw new RuntimeException('untrained model');
        }
        
        if ($length < $this->n) {
            throw new InvalidArgumentException(sprintf('length %d must be at least n %d', $length, $this->n));
        }
        
        $keys = array_keys($this->model);
        $seed = '';
        
        if ($startWith !== null) {
            $startLength = mb_strlen($startWith);
            if ($startLength >= $this->n) {
                $seed = mb_substr($startWith, 0, $this->n);
            }
        }
        
        if ($seed === '' || !isset($this->model[$seed])) {
            $seed = $keys[$this->secureRandInt(count($keys))];
            if ($startWith !== null) {
                $this->log('Warning: Starting text %s not found, using random n-gram', $startWith);
            }
        } else {
            $this->log('Starting generation with: %s', $seed);
        }
        
        $output = $seed;
        
        while (mb_strlen($output) < $length) {
            $currentSeed = mb_substr($output, -$this->n);
            $nextChars = $this->model[$currentSeed] ?? [];
            
            if (count($nextChars) === 0) {
                $similar = $this->findSimilarNgram($currentSeed);
                if ($similar !== '' && isset($this->model[$similar])) {
                    $this->log('Fallback: using similar n-gram %s for %s', $similar, $currentSeed);
                    $nextChars = $this->model[$similar];
                } else {
                    if (empty($this->text)) {
                        throw new RuntimeException('no text available for fallback');
                    }
                    $textLength = mb_strlen($this->text);
                    $randomPos = $this->secureRandInt($textLength);
                    $nextChar = mb_substr($this->text, $randomPos, 1);
                    $output .= $nextChar;
                    continue;
                }
            }
            
            if (count($nextChars) === 0) {
                throw new RuntimeException('no valid transitions available');
            }
            
            $nextChar = $nextChars[$this->secureRandInt(count($nextChars))];
            $output .= $nextChar;
        }
        
        return mb_substr($output, 0, $length);
    }

    public function findSimilarNgram(string $target): string
    {
        $bestMatch = '';
        $bestDistance = PHP_INT_MAX;
        $targetLength = mb_strlen($target);
        
        foreach ($this->model as $key => $transitions) {
            if (count($transitions) === 0) {
                continue;
            }
            
            $keyLength = mb_strlen($key);
            if ($keyLength !== $targetLength) {
                continue;
            }
            
            $distance = $this->simpleStringDistance($target, $key);
            if ($distance < $bestDistance) {
                $bestDistance = $distance;
                $bestMatch = $key;
            }
            
            if ($bestDistance === 0) {
                break;
            }
        }
        
        return $bestMatch;
    }

    private function simpleStringDistance(string $a, string $b): int
    {
        $distance = 0;
        $length = mb_strlen($a);
        for ($i = 0; $i < $length; $i++) {
            if (mb_substr($a, $i, 1) !== mb_substr($b, $i, 1)) {
                $distance++;
            }
        }
        return $distance;
    }

    public function validateModel(): void
    {
        if ($this->n <= 0) {
            throw new RuntimeException(sprintf('invalid n value: %d', $this->n));
        }
        
        foreach ($this->model as $key => $transitions) {
            $len = mb_strlen($key);
            if ($len !== $this->n) {
                throw new RuntimeException(sprintf('invalid key length: %s (expected %d)', $key, $this->n));
            }
            if (!is_array($transitions)) {
                throw new RuntimeException(sprintf('invalid transitions type for key %s', $key));
            }
        }
    }

    public function getModelStats(): ModelStats
    {
        $stats = new ModelStats();
        
        foreach ($this->model as $transitions) {
            $count = count($transitions);
            $stats->nGrams++;
            $stats->totalTransitions += $count;
            
            if ($count > $stats->maxTransitions) {
                $stats->maxTransitions = $count;
            }
            if ($stats->minTransitions === -1 || $count < $stats->minTransitions) {
                $stats->minTransitions = $count;
            }
            if ($count === 0) {
                $stats->deadEnds++;
            }
        }
        
        if ($stats->nGrams > 0) {
            $stats->avgTransitions = (float)($stats->totalTransitions / $stats->nGrams);
        }
        
        return $stats;
    }

    public function saveModel(string $filename): void
    {
        $payload = [
            'n' => $this->n,
            'model' => $this->model,
            'meta' => [
                'timestamp' => (new DateTimeImmutable())->format(DateTime::ATOM),
                'size' => count($this->model),
            ],
        ];
        
        $json = json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
        if ($json === false) {
            throw new RuntimeException('failed to encode model to JSON');
        }
        
        $result = file_put_contents($filename, $json);
        if ($result === false) {
            throw new RuntimeException(sprintf('failed to write model file: %s', $filename));
        }
        
        $this->log('Model saved to %s', $filename);
    }

    public function loadModel(string $filename): void
    {
        if (!is_readable($filename)) {
            throw new RuntimeException(sprintf('failed to open model file: %s', $filename));
        }
        
        $json = file_get_contents($filename);
        if ($json === false) {
            throw new RuntimeException(sprintf('failed to read model file: %s', $filename));
        }
        
        $data = json_decode($json, true);
        if (!is_array($data) || !isset($data['n']) || !isset($data['model'])) {
            throw new RuntimeException('failed to decode model');
        }
        
        $loadedN = (int) $data['n'];
        if ($loadedN !== $this->n) {
            throw new RuntimeException(sprintf(
                'loaded model n=%d does not match current n=%d', 
                $loadedN, 
                $this->n
            ));
        }
        
        $this->model = $data['model'];
        $this->log('Model loaded from %s with %d n-grams', $filename, count($this->model));
    }

    public function getAvailableKeys(): array
    {
        return array_keys($this->model);
    }

    public function getTransitions(string $key): array
    {
        return $this->model[$key] ?? [];
    }

    public function reset(): void
    {
        $this->model = [];
        $this->text = '';
        $this->clearLogs();
    }

    public function summary(): string
    {
        $stats = $this->getModelStats();
        return sprintf(
            "Model Statistics:\n- N-Grams: %d\n- Total Transitions: %d\n- Average Transitions: %.2f\n- Max Transitions: %d\n- Min Transitions: %d\n- Dead Ends: %d\n",
            $stats->nGrams,
            $stats->totalTransitions,
            $stats->avgTransitions,
            $stats->maxTransitions,
            $stats->minTransitions,
            $stats->deadEnds
        );
    }
}

function main(): void
{
    $trainingText = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>/?'
        . 'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.';

    $markov = new MarkovSeedGenerator(3, true, true);

    try {
        $markov->train($trainingText);
        
        try {
            $markov->validateModel();
        } catch (Throwable $e) {
            fwrite(STDERR, 'Model validation warning: ' . $e->getMessage() . PHP_EOL);
        }
        
        echo $markov->summary();

        for ($i = 0; $i < 5; $i++) {
            $gen = $markov->generate(16);
            printf("Generated %d: %s\n", $i + 1, $gen);
        }

        echo PHP_EOL;
        $seeded = $markov->generate(20, 'The');
        printf("Seeded generation: %s\n", $seeded);

        $markov->saveModel('markov_model.json');

        $markov2 = new MarkovSeedGenerator(3, true, true);
        $markov2->loadModel('markov_model.json');
        $reloaded = $markov2->generate(16);
        printf("From reloaded model: %s\n", $reloaded);

    } catch (Throwable $e) {
        fwrite(STDERR, 'Error: ' . $e->getMessage() . PHP_EOL);
        exit(1);
    }
}

main();
