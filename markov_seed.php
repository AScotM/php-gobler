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
    public int $uniqueTransitions = 0;
    public float $entropy = 0.0;

    public function toArray(): array
    {
        return [
            'nGrams' => $this->nGrams,
            'totalTransitions' => $this->totalTransitions,
            'avgTransitions' => $this->avgTransitions,
            'maxTransitions' => $this->maxTransitions,
            'minTransitions' => $this->minTransitions,
            'deadEnds' => $this->deadEnds,
            'uniqueTransitions' => $this->uniqueTransitions,
            'entropy' => $this->entropy,
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
    private int $maxModelSize;

    public function __construct(int $n = 3, bool $verbose = false, bool $useSecureRand = true, int $maxModelSize = 100000)
    {
        if ($n <= 0) {
            throw new InvalidArgumentException('n must be positive');
        }
        if ($maxModelSize <= 0) {
            throw new InvalidArgumentException('maxModelSize must be positive');
        }
        
        $this->n = $n;
        $this->model = [];
        $this->text = '';
        $this->verbose = $verbose;
        $this->logMessages = [];
        $this->useSecureRand = $useSecureRand;
        $this->maxModelSize = $maxModelSize;
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
        
        // Bias-free rejection sampling fallback
        $bits = ceil(log($n, 2)) + 8;
        $bytes = (int)ceil($bits / 8);
        
        do {
            $randomBytes = random_bytes($bytes);
            $value = 0;
            for ($i = 0; $i < $bytes; $i++) {
                $value = ($value << 8) | ord($randomBytes[$i]);
            }
            $value &= (1 << $bits) - 1;
        } while ($value >= $n);
        
        return $value;
    }

    private function sanitizeText(string $text): string
    {
        $text = preg_replace('/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/u', '', $text);
        return trim($text);
    }

    private function normalizeWhitespace(string $text): string
    {
        return preg_replace('/\s+/u', ' ', $text);
    }

    public function train(string $inputText): void
    {
        $text = $this->sanitizeText($inputText);
        $text = $this->normalizeWhitespace($text);
        $textLength = mb_strlen($text);
        
        if ($textLength <= $this->n) {
            throw new InvalidArgumentException(sprintf('text length %d must be greater than n %d', $textLength, $this->n));
        }
        
        $this->text = $text;
        $this->model = [];
        
        $startTime = microtime(true);
        
        for ($i = 0; $i <= $textLength - $this->n - 1; $i++) {
            $key = mb_substr($text, $i, $this->n);
            $nextChar = mb_substr($text, $i + $this->n, 1);
            
            if (!isset($this->model[$key])) {
                if (count($this->model) >= $this->maxModelSize) {
                    $this->log('Warning: Model size limit reached (%d entries)', $this->maxModelSize);
                    break;
                }
                $this->model[$key] = ['count' => 0, 'chars' => []];
            }
            
            // Ensure character is stored as string
            $nextCharStr = (string)$nextChar;
            if (!isset($this->model[$key]['chars'][$nextCharStr])) {
                $this->model[$key]['chars'][$nextCharStr] = 0;
            }
            $this->model[$key]['chars'][$nextCharStr]++;
            $this->model[$key]['count']++;
        }
        
        $trainingTime = microtime(true) - $startTime;
        $this->log('Trained model with %d n-grams in %.3f seconds', count($this->model), $trainingTime);
    }

    public function trainFromFile(string $filename): void
    {
        if (!is_string($filename) || $filename === '') {
            throw new InvalidArgumentException('filename must be a non-empty string');
        }
        
        $realpath = realpath($filename);
        if ($realpath === false) {
            throw new RuntimeException(sprintf('file does not exist or cannot be resolved: %s', $filename));
        }
        
        if (!is_readable($realpath)) {
            throw new RuntimeException(sprintf('file not readable: %s', $realpath));
        }
        
        $size = filesize($realpath);
        if ($size === false) {
            throw new RuntimeException(sprintf('failed to get file size: %s', $realpath));
        }
        
        if ($size === 0) {
            throw new RuntimeException('training file is empty');
        }
        
        $maxSize = 100 * 1024 * 1024;
        if ($size > $maxSize) {
            throw new RuntimeException(sprintf('file too large: %d bytes (max: %d)', $size, $maxSize));
        }
        
        $this->log('Training from file: %s (%d bytes)', $realpath, $size);
        
        $content = file_get_contents($realpath);
        if ($content === false) {
            throw new RuntimeException(sprintf('failed to read training file: %s', $realpath));
        }
        
        $this->train($content);
    }

    private function weightedRandomChoice(array $charCounts): string
    {
        $total = 0;
        foreach ($charCounts as $count) {
            $total += $count;
        }
        
        if ($total <= 0) {
            return '';
        }
        
        $rand = $this->secureRandInt($total);
        $cumulative = 0;
        
        foreach ($charCounts as $char => $count) {
            $cumulative += $count;
            if ($rand < $cumulative) {
                return (string)$char;
            }
        }
        
        return '';
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
        
        if ($startWith !== null && $startWith !== '') {
            $startLength = mb_strlen($startWith);
            if ($startLength >= $this->n) {
                $seed = mb_substr($startWith, 0, $this->n);
            }
        }
        
        if ($seed === '' || !isset($this->model[$seed])) {
            $seed = $keys[$this->secureRandInt(count($keys))];
            if ($startWith !== null && $startWith !== '') {
                $this->log('Starting text "%s" not found in model, using random n-gram: "%s"', $startWith, $seed);
            }
        } else {
            $this->log('Starting generation with: "%s"', $seed);
        }
        
        $output = $seed;
        
        while (mb_strlen($output) < $length) {
            $currentSeed = mb_substr($output, -$this->n);
            
            if (!isset($this->model[$currentSeed])) {
                $similar = $this->findSimilarNgram($currentSeed);
                if ($similar !== '' && isset($this->model[$similar])) {
                    $this->log('Fallback: using similar n-gram "%s" for "%s"', $similar, $currentSeed);
                    $currentSeed = $similar;
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
            
            $charCounts = $this->model[$currentSeed]['chars'] ?? [];
            if (empty($charCounts)) {
                throw new RuntimeException(sprintf('no valid transitions available for n-gram "%s"', $currentSeed));
            }
            
            $nextChar = $this->weightedRandomChoice($charCounts);
            if ($nextChar === '') {
                throw new RuntimeException('failed to select next character');
            }
            
            $output .= $nextChar;
        }
        
        return mb_substr($output, 0, $length);
    }

    public function findSimilarNgram(string $target): string
    {
        $bestMatch = '';
        $bestDistance = PHP_INT_MAX;
        $targetLength = mb_strlen($target);
        
        if ($targetLength !== $this->n) {
            return $bestMatch;
        }
        
        foreach ($this->model as $key => $data) {
            if (empty($data['chars'])) {
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
        $lengthA = mb_strlen($a);
        $lengthB = mb_strlen($b);
        
        if ($lengthA !== $lengthB) {
            return PHP_INT_MAX;
        }
        
        $distance = 0;
        for ($i = 0; $i < $lengthA; $i++) {
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
        
        foreach ($this->model as $key => $data) {
            $len = mb_strlen($key);
            if ($len !== $this->n) {
                throw new RuntimeException(sprintf('invalid key length: %s (expected %d)', $key, $this->n));
            }
            if (!is_array($data) || !isset($data['count']) || !isset($data['chars'])) {
                throw new RuntimeException(sprintf('invalid data structure for key %s', $key));
            }
            if ($data['count'] <= 0) {
                throw new RuntimeException(sprintf('non-positive transition count for key %s', $key));
            }
        }
    }

    public function getModelStats(): ModelStats
    {
        $stats = new ModelStats();
        
        foreach ($this->model as $data) {
            $count = $data['count'];
            $unique = count($data['chars']);
            
            $stats->nGrams++;
            $stats->totalTransitions += $count;
            $stats->uniqueTransitions += $unique;
            
            if ($count > $stats->maxTransitions) {
                $stats->maxTransitions = $count;
            }
            if ($stats->minTransitions === -1 || $count < $stats->minTransitions) {
                $stats->minTransitions = $count;
            }
            if ($unique === 0) {
                $stats->deadEnds++;
            }
            
            if ($unique > 0) {
                $entropy = 0.0;
                foreach ($data['chars'] as $charCount) {
                    $probability = $charCount / $count;
                    $entropy -= $probability * log($probability, 2);
                }
                $stats->entropy += $entropy;
            }
        }
        
        if ($stats->nGrams > 0) {
            $stats->avgTransitions = (float)($stats->totalTransitions / $stats->nGrams);
            $stats->entropy /= $stats->nGrams;
        }
        
        return $stats;
    }

    public function saveModel(string $filename): void
    {
        if (!is_string($filename) || $filename === '') {
            throw new InvalidArgumentException('filename must be a non-empty string');
        }
        
        $dir = dirname($filename);
        if ($dir !== '' && !is_dir($dir)) {
            throw new RuntimeException(sprintf('directory does not exist: %s', $dir));
        }
        
        $payload = [
            'n' => $this->n,
            'model' => $this->model,
            'meta' => [
                'timestamp' => (new DateTimeImmutable())->format(DateTime::ATOM),
                'size' => count($this->model),
                'version' => '1.1',
            ],
        ];
        
        $json = json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_THROW_ON_ERROR);
        
        $tempFile = tempnam($dir ?: sys_get_temp_dir(), 'markov_');
        if ($tempFile === false) {
            throw new RuntimeException('failed to create temporary file');
        }
        
        try {
            if (file_put_contents($tempFile, $json) === false) {
                throw new RuntimeException('failed to write temporary file');
            }
            
            if (!rename($tempFile, $filename)) {
                throw new RuntimeException(sprintf('failed to move file to destination: %s', $filename));
            }
            
            $this->log('Model saved to %s', $filename);
        } catch (Throwable $e) {
            if (file_exists($tempFile)) {
                unlink($tempFile);
            }
            throw $e;
        }
    }

    public function loadModel(string $filename): void
    {
        if (!is_string($filename) || $filename === '') {
            throw new InvalidArgumentException('filename must be a non-empty string');
        }
        
        $realpath = realpath($filename);
        if ($realpath === false) {
            throw new RuntimeException(sprintf('file does not exist: %s', $filename));
        }
        
        if (!is_readable($realpath)) {
            throw new RuntimeException(sprintf('file not readable: %s', $realpath));
        }
        
        $size = filesize($realpath);
        if ($size === false || $size === 0) {
            throw new RuntimeException('file is empty or invalid');
        }
        
        $json = file_get_contents($realpath);
        if ($json === false) {
            throw new RuntimeException(sprintf('failed to read model file: %s', $realpath));
        }
        
        try {
            $data = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
        } catch (JsonException $e) {
            throw new RuntimeException('failed to decode JSON: ' . $e->getMessage());
        }
        
        if (!is_array($data) || !isset($data['n']) || !isset($data['model'])) {
            throw new RuntimeException('invalid model structure');
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
        $this->log('Model loaded from %s with %d n-grams', $realpath, count($this->model));
    }

    public function getAvailableKeys(): array
    {
        return array_keys($this->model);
    }

    public function getTransitions(string $key): array
    {
        if (!isset($this->model[$key])) {
            return [];
        }
        return $this->model[$key]['chars'];
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
            "Model Statistics:\n- N-Grams: %d\n- Total Transitions: %d\n- Unique Transitions: %d\n- Average Transitions: %.2f\n- Max Transitions: %d\n- Min Transitions: %d\n- Dead Ends: %d\n- Average Entropy: %.3f\n",
            $stats->nGrams,
            $stats->totalTransitions,
            $stats->uniqueTransitions,
            $stats->avgTransitions,
            $stats->maxTransitions,
            $stats->minTransitions,
            $stats->deadEnds,
            $stats->entropy
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
