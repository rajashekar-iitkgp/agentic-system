import json
from typing import Dict, Any

class SchemaCompressor:
    
    @staticmethod
    def compress_tool_definition(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        clean_desc = (description[:197] + '...') if len(description) > 200 else description
        
        compressed_schema = SchemaCompressor._minify_schema(schema)
        
        return {
            "n": name,
            "d": clean_desc,
            "s": compressed_schema
        }

    @staticmethod
    def _minify_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        minified = {}
        target_keys = ["type", "properties", "required", "enum", "items"]
        
        for k, v in schema.items():
            if k in target_keys:
                if k == "properties":
                    minified[k] = {prop: SchemaCompressor._minify_schema(details) for prop, details in v.items()}
                elif k == "items":
                    minified[k] = SchemaCompressor._minify_schema(v)
                else:
                    minified[k] = v
        return minified

schema_compressor = SchemaCompressor()
