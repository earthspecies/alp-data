import argparse
import io
import json
import os
import sys

import ijson
from tqdm import tqdm


class JSONLStreamProcessor:
    def __init__(self, input_file, output_dir, chunk_size=10000, buffer_size=10 * 1024 * 1024, verbose=False):
        """
        Initialize the JSONL stream processor

        Args:
            input_file (str): Path to the input JSONL file
            output_dir (str): Directory to save output chunks
            chunk_size (int): Number of objects per output file
            buffer_size (int): Size of the buffer in bytes
            verbose (bool): Whether to print detailed information
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.verbose = verbose

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Statistics
        self.valid_count = 0
        self.invalid_count = 0
        self.chunk_count = 0

    def process(self):
        """Process the JSONL file in stream mode"""
        valid_objects = []

        # Get file size for progress bar
        file_size = os.path.getsize(self.input_file)
        bytes_processed = 0

        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Processing JSONL") as pbar:
            # Process the file in chunks to handle large files
            with open(self.input_file, "rb") as f:
                buffer = b""

                while True:
                    chunk = f.read(self.buffer_size)
                    if not chunk:
                        break

                    bytes_processed += len(chunk)
                    pbar.update(len(chunk))

                    buffer += chunk

                    # Process complete lines in the buffer
                    lines = buffer.split(b"\n")

                    # Keep the last (potentially incomplete) line in the buffer
                    buffer = lines[-1]

                    # Process complete lines
                    for line in lines[:-1]:
                        if not line.strip():
                            continue

                        try:
                            obj = self._parse_json_line(line)
                            if obj is not None:
                                valid_objects.append(obj)
                                self.valid_count += 1

                                # If we've reached chunk_size objects, write to a file
                                if len(valid_objects) >= self.chunk_size:
                                    self._write_chunk(valid_objects)
                                    self.chunk_count += 1
                                    valid_objects = []
                            else:
                                self.invalid_count += 1
                        except Exception as e:
                            self.invalid_count += 1
                            if self.verbose:
                                print(f"Error processing line: {e}")
                                print(f"Problematic line: {line[:200]}..." if len(line) > 200 else line)

                # Process the final buffer if it contains data
                if buffer.strip():
                    try:
                        obj = self._parse_json_line(buffer)
                        if obj is not None:
                            valid_objects.append(obj)
                            self.valid_count += 1
                        else:
                            self.invalid_count += 1
                    except Exception as e:
                        self.invalid_count += 1
                        if self.verbose:
                            print(f"Error processing final buffer: {e}")

        # Write any remaining objects
        if valid_objects:
            self._write_chunk(valid_objects)
            self.chunk_count += 1

        return self.valid_count, self.invalid_count, self.chunk_count

    def _parse_json_line(self, line):
        """Parse a single line as JSON with robust error handling"""
        if not line.strip():
            return None

        try:
            # Standard JSON parsing
            return json.loads(line)
        except json.JSONDecodeError:
            # Try more lenient parsing approaches
            try:
                # Try using ijson for more robust parsing
                line_io = io.BytesIO(line)
                parser = ijson.parse(line_io)

                # Use the first complete object found
                for prefix, event, value in parser:
                    if prefix == "" and event == "map_key":
                        # Beginning of an object
                        parsed_obj = {}
                        current_key = value
                    elif prefix == "":
                        # Direct value or end of parsing
                        return value
                    elif "." not in prefix:
                        # Top-level keys
                        if parsed_obj is not None:
                            parsed_obj[current_key] = value

                return parsed_obj
            except Exception:
                # If all parsing fails, try to extract a JSON-like structure
                try:
                    line_str = line.decode("utf-8")
                    # Find outermost brackets
                    if "{" in line_str and "}" in line_str:
                        start = line_str.find("{")
                        end = line_str.rfind("}") + 1
                        maybe_json = line_str[start:end]
                        return json.loads(maybe_json)
                    elif "[" in line_str and "]" in line_str:
                        start = line_str.find("[")
                        end = line_str.rfind("]") + 1
                        maybe_json = line_str[start:end]
                        return json.loads(maybe_json)
                except Exception:
                    # Final fallback: try to fix common JSON errors
                    try:
                        line_str = line.decode("utf-8").strip()
                        # Fix trailing commas
                        line_str = line_str.replace(",}", "}").replace(",]", "]")
                        # Ensure proper quotes for keys
                        line_str = self._fix_quotes(line_str)
                        return json.loads(line_str)
                    except Exception:
                        pass

        return None

    def _fix_quotes(self, text):
        """Fix common quote issues in JSON"""
        # Replace single quotes with double quotes
        result = ""
        in_string = False
        escape = False

        for char in text:
            if char == "\\" and not escape:
                escape = True
                result += char
                continue

            if char == '"' and not escape:
                in_string = not in_string

            if char == "'" and not in_string and not escape:
                result += '"'
            else:
                result += char

            escape = False

        return result

    def _write_chunk(self, objects):
        """Writes a list of JSON objects to a JSONL file"""
        output_file = os.path.join(self.output_dir, f"chunk_{self.chunk_count:04d}.jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            for obj in objects:
                try:
                    f.write(json.dumps(obj) + "\n")
                except Exception as e:
                    if self.verbose:
                        print(f"Error writing object: {e}")
                    self.invalid_count += 1
                    self.valid_count -= 1


def main():
    parser = argparse.ArgumentParser(description="Stream processor for large and corrupted JSONL files")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_dir", help="Directory to save the output chunks")
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Number of JSON objects per output file (default: 10000)"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10 * 1024 * 1024, help="Size of the buffer in bytes (default: 10MB)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed information about processing")

    args = parser.parse_args()

    print(f"Processing file: {args.input_file}")

    try:
        processor = JSONLStreamProcessor(
            args.input_file, args.output_dir, args.chunk_size, args.buffer_size, args.verbose
        )

        valid_count, invalid_count, chunk_count = processor.process()

        print("\nProcessing complete!")
        print(f"Valid JSON objects: {valid_count}")
        print(f"Invalid JSON objects: {invalid_count}")
        print(f"Output chunks created: {chunk_count}")
        print(f"Output directory: {args.output_dir}")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Partial results have been saved.")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
