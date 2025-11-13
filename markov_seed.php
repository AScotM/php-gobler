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
                // fall through to deterministic fallback
            }
        }
        return (int) (abs(crc32((string) getmypid() ^ (string) $n)) % $n);
    }

    private function sanitizeText(string $text): string
    {
        $chars = preg_split('//u', $text, -1, PREG_SPLIT_NO_EMPTY);
        if ($chars === false) {
            return '';
        }
        $out = [];
        foreach ($chars as $ch) {
            $ord = $this->unicodeOrd($ch);
            if ($ord === null) {
                continue;
            }
            if ($ord === 9 || $ord === 10 || $ord === 13) {
                $out[] = $ch;
                continue;
            }
            if ($this->isControl($ord)) {
                continue;
            }
            $out[] = $ch;
        }
        return implode('', $out);
    }

    private function isControl(int $codePoint): bool
    {
        return ($codePoint < 32) || ($codePoint === 127);
    }

    private function unicodeOrd(string $char): ?int
    {
        $u = mb_convert_encoding($char, 'UCS-4BE', 'UTF-8');
        if ($u === false || $u === '') {
            return null;
        }
        $val = unpack('N', $u);
        if (!is_array($val) || !isset($val[1])) {
            return null;
        }
        return $val[1];
    }

    public function train(string $inputText): void
    {
        $text = $this->sanitizeText($inputText);
        $runes = preg_split('//u', $text, -1, PREG_SPLIT_NO_EMPTY);
        if ($runes === false) {
            throw new RuntimeException('failed to split text into characters');
        }
        if (count($runes) <= $this->n) {
            throw new InvalidArgumentException(sprintf('text length %d must be greater than n %d', count($runes), $this->n));
        }
        $this->text = $text;
        $limit = count($runes) - $this->n;
        for ($i = 0; $i < $limit; $i++) {
            $key = implode('', array_slice($runes, $i, $this->n));
            $nextChar = $runes[$i + $this->n];
            if (!array_key_exists($key, $this->model)) {
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
        $info = stat($filename);
        if ($info === false) {
            throw new RuntimeException(sprintf('failed to get file info: %s', $filename));
        }
        $size = $info['size'] ?? 0;
        if ($size === 0) {
            throw new RuntimeException('training file is empty');
        }
        $maxSize = 100 * 1024 * 1024;
        if ($size > $maxSize) {
            throw new RuntimeException(sprintf('file too large: %d bytes', $size));
        }
        $this->log('Training from file: %s', $filename);
        $handle = fopen($filename, 'rb');
        if ($handle === false) {
            throw new RuntimeException(sprintf('failed to open training file: %s', $filename));
        }
        $bufferSize = 8192;
        $processed = 0;
        $chunks = '';
        while (!feof($handle)) {
            $data = fread($handle, $bufferSize);
            if ($data === false) {
                fclose($handle);
                throw new RuntimeException('error reading file');
            }
            $n = strlen($data);
            if ($n === 0) {
                break;
            }
            $chunk = $this->sanitizeText($data);
            $chunks .= $chunk;
            $processed += $n;
            if ($this->verbose && $size > 0) {
                $percent = ($processed / $size) * 100.0;
                $this->log('Processed %d/%d bytes (%.1f%%)', $processed, $size, $percent);
            }
        }
        fclose($handle);
        $this->train($chunks);
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
        if ($startWith !== null && mb_strlen($startWith) >= $this->n) {
            $seed = mb_substr($startWith, 0, $this->n);
        }
        if ($seed === '' || !array_key_exists($seed, $this->model)) {
            $seed = $keys[$this->secureRandInt(count($keys))];
            if ($startWith !== null) {
                $this->log('Warning: Starting text %s not found, using random n-gram', $startWith);
            }
        } else {
            $this->log('Starting generation with: %s', $seed);
        }
        $output = preg_split('//u', $seed, -1, PREG_SPLIT_NO_EMPTY);
        if ($output === false) {
            throw new RuntimeException('failed to split seed');
        }
        while (count($output) < $length) {
            $seedKey = implode('', array_slice($output, count($output) - $this->n, $this->n));
            $nextChars = $this->model[$seedKey] ?? [];
            if (count($nextChars) === 0) {
                $similar = $this->findSimilarNgram($seedKey);
                if ($similar !== '') {
                    $this->log('Fallback: using similar n-gram %s for %s', $similar, $seedKey);
                    $nextChars = $this->model[$similar] ?? [];
                } else {
                    $runes = preg_split('//u', $this->text, -1, PREG_SPLIT_NO_EMPTY);
                    if ($runes === false || count($runes) === 0) {
                        throw new RuntimeException('no text available for fallback');
                    }
                    $nextChar = $runes[$this->secureRandInt(count($runes))];
                    $output[] = $nextChar;
                    continue;
                }
            }
            if (count($nextChars) === 0) {
                throw new RuntimeException('no valid transitions available');
            }
            $nextChar = $nextChars[$this->secureRandInt(count($nextChars))];
            $output[] = $nextChar;
        }
        return implode('', array_slice($output, 0, $length));
    }

    public function findSimilarNgram(string $target): string
    {
        $bestMatch = '';
        $bestDistance = -1;
        $targetRunes = preg_split('//u', $target, -1, PREG_SPLIT_NO_EMPTY);
        if ($targetRunes === false) {
            return '';
        }
        foreach ($this->model as $key => $transitions) {
            if (count($transitions) === 0) {
                continue;
            }
            $keyRunes = preg_split('//u', $key, -1, PREG_SPLIT_NO_EMPTY);
            if ($keyRunes === false) {
                continue;
            }
            $distance = $this->levenshteinDistance($targetRunes, $keyRunes);
            if ($bestDistance === -1 || $distance < $bestDistance) {
                $bestDistance = $distance;
                $bestMatch = $key;
            }
            if ($bestDistance <= 1) {
                break;
            }
        }
        return $bestMatch;
    }

    private function levenshteinDistance(array $a, array $b): int
    {
        $la = count($a);
        $lb = count($b);
        if ($la === 0) {
            return $lb;
        }
        if ($lb === 0) {
            return $la;
        }
        $matrix = [];
        for ($i = 0; $i <= $la; $i++) {
            $matrix[$i] = array_fill(0, $lb + 1, 0);
            $matrix[$i][0] = $i;
        }
        for ($j = 0; $j <= $lb; $j++) {
            $matrix[0][$j] = $j;
        }
        for ($i = 1; $i <= $la; $i++) {
            for ($j = 1; $j <= $lb; $j++) {
                $cost = ($a[$i - 1] === $b[$j - 1]) ? 0 : 1;
                $matrix[$i][$j] = $this->min3(
                    $matrix[$i - 1][$j] + 1,
                    $matrix[$i][$j - 1] + 1,
                    $matrix[$i - 1][$j - 1] + $cost
                );
            }
        }
        return $matrix[$la][$lb];
    }

    private function min3(int $a, int $b, int $c): int
    {
        if ($a < $b && $a < $c) {
            return $a;
        }
        if ($b < $c) {
            return $b;
        }
        return $c;
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
            if (count($transitions) === 0) {
                throw new RuntimeException(sprintf('key %s has no transitions', $key));
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
            $stats->avgTransitions = $stats->totalTransitions / $stats->nGrams;
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
        $this->n = (int) $data['n'];
        $this->model = [];
        foreach ($data['model'] as $k => $arr) {
            if (!is_array($arr)) {
                continue;
            }
            $this->model[$k] = array_values($arr);
        }
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
        return sprintf("Model Statistics:\n- N-Grams: %d\n- Total Transitions: %d\n- Average Transitions: %.2f\n- Max Transitions: %d\n- Min Transitions: %d\n- Dead Ends: %d\n",
            $stats->nGrams, $stats->totalTransitions, $stats->avgTransitions, $stats->maxTransitions, $stats->minTransitions, $stats->deadEnds);
    }
}

function main(): void
{
    $trainingText = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>/?' .
        'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.';

    $markov = new MarkovSeedGenerator(3, true, true);

    try {
        $markov->train($trainingText);
    } catch (Throwable $e) {
        fwrite(STDERR, 'Training error: ' . $e->getMessage() . PHP_EOL);
        exit(1);
    }

    try {
        $markov->validateModel();
    } catch (Throwable $e) {
        fwrite(STDERR, 'Model validation warning: ' . $e->getMessage() . PHP_EOL);
    }

    echo $markov->summary();

    for ($i = 0; $i < 5; $i++) {
        try {
            $gen = $markov->generate(16);
            printf("Generated %d: %s\n", $i + 1, $gen);
        } catch (Throwable $e) {
            fwrite(STDERR, 'Error: ' . $e->getMessage() . PHP_EOL);
        }
    }

    echo PHP_EOL;

    try {
        $seeded = $markov->generate(20, 'The');
        printf("Seeded generation: %s\n", $seeded);
    } catch (Throwable $e) {
        fwrite(STDERR, 'Error: ' . $e->getMessage() . PHP_EOL);
    }

    try {
        $markov->saveModel('markov_model.json');
    } catch (Throwable $e) {
        fwrite(STDERR, 'Error saving model: ' . $e->getMessage() . PHP_EOL);
    }

    $markov2 = new MarkovSeedGenerator(3, true, true);

    try {
        $markov2->loadModel('markov_model.json');
        $reloaded = $markov2->generate(16);
        printf("From reloaded model: %s\n", $reloaded);
    } catch (Throwable $e) {
        fwrite(STDERR, 'Reload error: ' . $e->getMessage() . PHP_EOL);
    }
}

main();
