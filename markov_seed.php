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
    private int $n;
    private array $model;
    private string $text;
    private array $characters;
    private bool $verbose;
    private array $logMessages;
    private bool $useSecureRand;
    private int $maxModelSize;
    private int $maxGenerationLength;

    public function __construct(int $n = 3, bool $verbose = false, bool $useSecureRand = true, int $maxModelSize = 100000, int $maxGenerationLength = 10000)
    {
        if ($n <= 0) {
            throw new InvalidArgumentException('n must be positive');
        }
        if ($maxModelSize <= 0) {
            throw new InvalidArgumentException('maxModelSize must be positive');
        }
        if ($maxGenerationLength <= 0) {
            throw new InvalidArgumentException('maxGenerationLength must be positive');
        }
        
        $this->n = $n;
        $this->model = [];
        $this->text = '';
        $this->characters = [];
        $this->verbose = $verbose;
        $this->logMessages = [];
        $this->useSecureRand = $useSecureRand;
        $this->maxModelSize = $maxModelSize;
        $this->maxGenerationLength = $maxGenerationLength;
    }

    public function getN(): int
    {
        return $this->n;
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
            }
        }
        
        $bits = (int)ceil(log($n, 2)) + 8;
        $bytes = (int)ceil($bits / 8);
        $maxAttempts = 100;
        $attempts = 0;
        
        do {
            if ($attempts++ >= $maxAttempts) {
                return random_int(0, $n - 1);
            }
            
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
        $text = preg_replace('/[\x00-\x08\x0E-\x1F\x7F-\x9F]/u', '', $text);
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
        $this->characters = preg_split('//u', $text, -1, PREG_SPLIT_NO_EMPTY);
        $this->model = [];
        
        $startTime = microtime(true);
        
        for ($i = 0; $i < $textLength - $this->n; $i++) {
            $key = '';
            for ($j = 0; $j < $this->n; $j++) {
                $key .= $this->characters[$i + $j];
            }
            $nextChar = $this->characters[$i + $this->n];
            
            if (!isset($this->model[$key])) {
                if (count($this->model) >= $this->maxModelSize) {
                    $this->log('Warning: Model size limit reached (%d entries)', $this->maxModelSize);
                    break;
                }
                $this->model[$key] = ['count' => 0, 'chars' => []];
            }
            
            if (!isset($this->model[$key]['chars'][$nextChar])) {
                $this->model[$key]['chars'][$nextChar] = 0;
            }
            $this->model[$key]['chars'][$nextChar]++;
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
        
        if ($length > $this->maxGenerationLength) {
            throw new InvalidArgumentException(sprintf('length %d exceeds maximum generation length %d', $length, $this->maxGenerationLength));
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
        $outputChars = preg_split('//u', $seed, -1, PREG_SPLIT_NO_EMPTY);
        
        while (count($outputChars) < $length) {
            $currentSeed = '';
            for ($i = count($outputChars) - $this->n; $i < count($outputChars); $i++) {
                $currentSeed .= $outputChars[$i];
            }
            
            if (!isset($this->model[$currentSeed])) {
                $similar = $this->findSimilarNgram($currentSeed);
                if ($similar !== '' && isset($this->model[$similar])) {
                    $this->log('Fallback: using similar n-gram "%s" for "%s"', $similar, $currentSeed);
                    $currentSeed = $similar;
                } else {
                    if (empty($this->characters)) {
                        throw new RuntimeException('no characters available for fallback');
                    }
                    $randomPos = $this->secureRandInt(count($this->characters));
                    $nextChar = $this->characters[$randomPos];
                    $outputChars[] = $nextChar;
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
            
            $outputChars[] = $nextChar;
        }
        
        return implode('', array_slice($outputChars, 0, $length));
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
            if (!is_array($data) || empty($data['chars'])) {
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
        
        $charsA = preg_split('//u', $a, -1, PREG_SPLIT_NO_EMPTY);
        $charsB = preg_split('//u', $b, -1, PREG_SPLIT_NO_EMPTY);
        
        $distance = 0;
        for ($i = 0; $i < $lengthA; $i++) {
            if ($charsA[$i] !== $charsB[$i]) {
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
            if (!is_array($data) || !isset($data['count']) || !isset($data['chars']) || !is_array($data['chars'])) {
                throw new RuntimeException(sprintf('invalid data structure for key %s', $key));
            }
            if ($data['count'] <= 0) {
                throw new RuntimeException(sprintf('non-positive transition count for key %s', $key));
            }
            foreach ($data['chars'] as $char => $count) {
                if (!is_int($count) || $count <= 0) {
                    throw new RuntimeException(sprintf('invalid character count for key %s, char %s', $key, $char));
                }
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
                'version' => '1.2',
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

    public function saveModelBinary(string $filename): void
    {
        if (!is_string($filename) || $filename === '') {
            throw new InvalidArgumentException('filename must be a non-empty string');
        }
        
        $dir = dirname($filename);
        if ($dir !== '' && !is_dir($dir)) {
            throw new RuntimeException(sprintf('directory does not exist: %s', $dir));
        }
        
        $data = [
            'n' => $this->n,
            'model' => serialize($this->model)
        ];
        
        $compressed = gzcompress(serialize($data));
        if ($compressed === false) {
            throw new RuntimeException('failed to compress model data');
        }
        
        $tempFile = tempnam($dir ?: sys_get_temp_dir(), 'markov_bin_');
        if ($tempFile === false) {
            throw new RuntimeException('failed to create temporary file');
        }
        
        try {
            if (file_put_contents($tempFile, $compressed) === false) {
                throw new RuntimeException('failed to write temporary file');
            }
            
            if (!rename($tempFile, $filename)) {
                throw new RuntimeException(sprintf('failed to move file to destination: %s', $filename));
            }
            
            $this->log('Binary model saved to %s', $filename);
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
        
        $content = file_get_contents($realpath);
        if ($content === false) {
            throw new RuntimeException(sprintf('failed to read model file: %s', $realpath));
        }
        
        try {
            $data = json_decode($content, true, 512, JSON_THROW_ON_ERROR);
        } catch (JsonException $e) {
            try {
                $uncompressed = gzuncompress($content);
                if ($uncompressed === false) {
                    throw new RuntimeException('failed to decompress data');
                }
                $data = unserialize($uncompressed);
                if (!is_array($data)) {
                    throw new RuntimeException('invalid binary model structure');
                }
            } catch (Throwable $e2) {
                throw new RuntimeException('failed to decode model: ' . $e->getMessage());
            }
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
        
        if (!is_array($data['model'])) {
            throw new RuntimeException('model data is not an array');
        }
        
        foreach ($data['model'] as $key => $value) {
            if (!is_string($key) || !is_array($value) || !isset($value['count']) || !isset($value['chars']) || !is_array($value['chars'])) {
                throw new RuntimeException(sprintf('invalid entry in loaded model for key: %s', $key));
            }
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

    public function offsetExists($offset): bool
    {
        return isset($this->model[$offset]);
    }

    public function offsetGet($offset): ?array
    {
        return $this->model[$offset] ?? null;
    }

    public function reset(): void
    {
        $this->model = [];
        $this->text = '';
        $this->characters = [];
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

    $markov = new MarkovSeedGenerator(3, true, true, 100000, 10000);

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
        $markov->saveModelBinary('markov_model.bin');

        $markov2 = new MarkovSeedGenerator(3, true, true);
        $markov2->loadModel('markov_model.json');
        $reloaded = $markov2->generate(16);
        printf("From reloaded JSON model: %s\n", $reloaded);
        
        $markov3 = new MarkovSeedGenerator(3, false, true);
        $markov3->loadModel('markov_model.bin');
        $binaryReloaded = $markov3->generate(16);
        printf("From reloaded binary model: %s\n", $binaryReloaded);

    } catch (Throwable $e) {
        fwrite(STDERR, 'Error: ' . $e->getMessage() . PHP_EOL);
        exit(1);
    }
}

if (PHP_SAPI === 'cli') {
    main();
}
