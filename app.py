import os
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from tempfile import NamedTemporaryFile
import whisper
import torch
import base64
import json
import gc
import time

# ===== CONFIGURAÇÃO DE LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== CONFIGURAÇÕES GLOBAIS =====
PORT = int(os.getenv('PORT', 5000))
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB limit

# ===== INICIALIZAÇÃO =====
app = Flask(__name__)
whisper_model = None
model_load_time = None

def get_system_info():
    """Retorna informações do sistema"""
    return {
        'device': DEVICE,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'model': WHISPER_MODEL,
        'model_loaded': whisper_model is not None,
        'model_load_time': model_load_time.isoformat() if model_load_time else None,
        'python_version': os.sys.version,
        'memory_usage': get_memory_usage()
    }

def get_memory_usage():
    """Obtém uso de memória (se disponível)"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total': f"{memory.total / (1024**3):.2f}GB",
            'available': f"{memory.available / (1024**3):.2f}GB",
            'percent': f"{memory.percent}%"
        }
    except ImportError:
        return {"status": "psutil not available"}

def load_whisper_model():
    """Carrega o modelo Whisper de forma otimizada"""
    global whisper_model, model_load_time
    
    if whisper_model is None:
        logger.info(f"🚀 Carregando modelo Whisper '{WHISPER_MODEL}' no dispositivo '{DEVICE}'...")
        start_time = time.time()
        
        try:
            # Configurar torch para usar menos memória
            if DEVICE == "cpu":
                torch.set_num_threads(2)  # Limitar threads no CPU
            
            whisper_model = whisper.load_model(
                WHISPER_MODEL, 
                device=DEVICE,
                download_root=None  # Usar cache padrão
            )
            
            load_duration = time.time() - start_time
            model_load_time = datetime.now()
            
            logger.info(f"✅ Modelo carregado com sucesso em {load_duration:.2f}s!")
            
            # Limpeza de memória
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            logger.error(traceback.format_exc())
            raise
    
    return whisper_model

# ===== ROUTES =====

@app.route('/')
def home():
    """Endpoint principal com informações do serviço"""
    try:
        return {
            'service': 'Whisper Transcription API',
            'status': 'online',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'endpoints': {
                'health': '/health',
                'transcribe': '/transcribe [POST]',
                'info': '/info'
            },
            'model': WHISPER_MODEL,
            'device': DEVICE
        }
    except Exception as e:
        logger.error(f"Erro no endpoint home: {e}")
        return {'error': str(e)}, 500

@app.route('/health')
def health():
    """Health check para Railway"""
    try:
        # Verificar se o modelo pode ser carregado
        start_time = time.time()
        model = load_whisper_model()
        load_time = time.time() - start_time
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': WHISPER_MODEL,
                'loaded': model is not None,
                'load_time_seconds': round(load_time, 2)
            },
            'system': get_system_info()
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, 500

@app.route('/info')
def info():
    """Informações detalhadas do sistema"""
    return {
        'service': 'Whisper Transcription API',
        'system': get_system_info(),
        'limits': {
            'max_audio_size_mb': MAX_AUDIO_SIZE / (1024*1024),
            'supported_formats': ['wav', 'mp3', 'm4a', 'flac', 'ogg']
        },
        'timestamp': datetime.now().isoformat()
    }

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Endpoint principal de transcrição"""
    start_time = time.time()
    temp_path = None
    
    try:
        # ===== VALIDAÇÃO DO REQUEST =====
        if not request.json:
            return {'error': 'JSON payload obrigatório'}, 400
        
        data = request.json
        if 'audio' not in data:
            return {'error': 'Campo "audio" (base64) obrigatório'}, 400

        audio_base64 = data['audio']
        language = data.get('language', 'pt')
        
        logger.info(f"📥 Iniciando transcrição - Idioma: {language}")

        # ===== VALIDAÇÃO E DECODIFICAÇÃO DO ÁUDIO =====
        try:
            audio_data = base64.b64decode(audio_base64)
            audio_size = len(audio_data)
            
            if audio_size > MAX_AUDIO_SIZE:
                return {
                    'error': f'Arquivo muito grande. Máximo: {MAX_AUDIO_SIZE/(1024*1024)}MB'
                }, 400
                
            logger.info(f"📁 Áudio decodificado: {audio_size / 1024:.1f}KB")
            
        except Exception as e:
            logger.error(f"❌ Erro ao decodificar base64: {e}")
            return {'error': f'Erro ao decodificar base64: {str(e)}'}, 400

        # ===== SALVAR ARQUIVO TEMPORÁRIO =====
        with NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        logger.info(f"💾 Arquivo salvo em: {temp_path}")

        # ===== CARREGAR MODELO E TRANSCREVER =====
        try:
            model = load_whisper_model()
            
            logger.info("🎯 Iniciando transcrição...")
            transcription_start = time.time()
            
            # Configurações otimizadas para Railway
            result = model.transcribe(
                temp_path,
                language=language if language and language != 'auto' else None,
                fp16=False,  # Usar fp32 para compatibilidade
                verbose=False,
                word_timestamps=False,  # Economizar processamento
                condition_on_previous_text=False  # Economizar memória
            )

            transcription_time = time.time() - transcription_start
            transcription = result['text'].strip()
            detected_language = result.get('language', 'unknown')
            
            logger.info(f"✅ Transcrição concluída em {transcription_time:.2f}s")
            logger.info(f"📝 Resultado: {len(transcription)} caracteres")

            # ===== RESPOSTA DE SUCESSO =====
            total_time = time.time() - start_time
            
            response = {
                'success': True,
                'transcription': transcription,
                'metadata': {
                    'language': detected_language,
                    'model': WHISPER_MODEL,
                    'device': DEVICE,
                    'audio_size_kb': round(audio_size / 1024, 1),
                    'processing_time_seconds': round(transcription_time, 2),
                    'total_time_seconds': round(total_time, 2),
                    'character_count': len(transcription),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            return response

        except Exception as e:
            logger.error(f"❌ Erro na transcrição: {e}")
            logger.error(traceback.format_exc())
            return {'error': f'Erro na transcrição: {str(e)}'}, 500

    except Exception as e:
        logger.error(f"❌ Erro geral na transcrição: {e}")
        logger.error(traceback.format_exc())
        return {'error': f'Erro interno: {str(e)}'}, 500

    finally:
        # ===== LIMPEZA =====
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"🗑️ Arquivo temporário removido: {temp_path}")
            except Exception as e:
                logger.warning(f"⚠️ Não foi possível remover arquivo temporário: {e}")
        
        # Limpeza de memória
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    return {
        'error': 'Endpoint não encontrado',
        'available_endpoints': ['/', '/health', '/info', '/transcribe'],
        'timestamp': datetime.now().isoformat()
    }, 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Erro interno do servidor: {error}")
    return {
        'error': 'Erro interno do servidor',
        'timestamp': datetime.now().isoformat()
    }, 500

@app.errorhandler(413)
def payload_too_large(error):
    return {
        'error': 'Payload muito grande',
        'max_size_mb': MAX_AUDIO_SIZE / (1024*1024),
        'timestamp': datetime.now().isoformat()
    }, 413

# ===== INICIALIZAÇÃO =====

if __name__ == '__main__':
    logger.info(f"🚀 Iniciando Whisper Transcription Service")
    logger.info(f"📡 Porta: {PORT}")
    logger.info(f"🤖 Modelo: {WHISPER_MODEL}")
    logger.info(f"💻 Dispositivo: {DEVICE}")
    
    # Pré-carregar modelo em desenvolvimento
    if os.getenv('FLASK_ENV') == 'development':
        try:
            load_whisper_model()
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível pré-carregar modelo: {e}")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
