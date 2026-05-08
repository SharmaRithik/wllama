import { useEffect, useMemo, useRef, useState } from 'react';
import type { PerfContextData } from '@reeselevine/wllama-webgpu';
import { useMessages } from '../utils/messages.context';
import { useWllama } from '../utils/wllama.context';
import { Message, Screen } from '../utils/types';
import { formatChat } from '../utils/utils';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faStop,
  faPaperclip,
  faXmark,
} from '@fortawesome/free-solid-svg-icons';
import ScreenWrapper from './ScreenWrapper';
import { useIntervalWhen } from '../utils/use-interval-when';
import { MarkdownMessage } from './MarkdownMessage';

const createInitialPerfData = (): PerfContextData => ({
  success: true,
  t_start_ms: 0,
  t_load_ms: 0,
  t_p_eval_ms: 0,
  t_eval_ms: 0,
  n_p_eval: 0,
  n_eval: 0,
  n_reused: 0,
});

export default function ChatScreen() {
  const [input, setInput] = useState('');
  const [perfData, setPerfData] = useState<PerfContextData>(
    createInitialPerfData()
  );
  const [perfError, setPerfError] = useState<string | null>(null);
  const [perfBusy, setPerfBusy] = useState(false);
  const [stagedImage, setStagedImage] = useState<{
    bytes: Uint8Array;
    mime: string;
    previewUrl: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const {
    currentConvId,
    isGenerating,
    createCompletion,
    createCompletionWithImage,
    navigateTo,
    loadedModel,
    getWllamaInstance,
    stopCompletion,
    currRuntimeInfo,
  } = useWllama();
  const visionAvailable = !!currRuntimeInfo?.hasVisionSupport;

  useEffect(() => {
    return () => {
      if (stagedImage?.previewUrl) URL.revokeObjectURL(stagedImage.previewUrl);
    };
  }, [stagedImage?.previewUrl]);
  const {
    getConversationById,
    addMessageToConversation,
    editMessageInConversation,
    newConversation,
  } = useMessages();

  useIntervalWhen(chatScrollToBottom, 500, isGenerating, true);

  const currConv = getConversationById(currentConvId);

  const refreshPerf = async () => {
    if (!loadedModel) return;
    setPerfBusy(true);
    setPerfError(null);
    try {
      setPerfData(await getWllamaInstance().getPerfContext());
    } catch (e) {
      setPerfError((e as any)?.message ?? 'Failed to fetch perf data');
    } finally {
      setPerfBusy(false);
    }
  };

  const resetPerf = async () => {
    if (!loadedModel) return;
    setPerfBusy(true);
    setPerfError(null);
    try {
      await getWllamaInstance().resetPerfContext();
      setPerfData(await getWllamaInstance().getPerfContext());
    } catch (e) {
      setPerfError((e as any)?.message ?? 'Failed to reset perf data');
    } finally {
      setPerfBusy(false);
    }
  };

  const formatTokPerSec = (tokens: number, ms: number) => {
    if (ms <= 0) return '0.0';
    return (tokens / (ms / 1000)).toFixed(1);
  };

  const onSubmit = async () => {
    if (isGenerating) return;
    if (!input.trim() && !stagedImage) return;

    const currHistory = currConv?.messages ?? [];
    const userInput = input;
    const sentImage = stagedImage;
    setInput('');
    setStagedImage(null);

    const userMsg: Message = {
      id: Date.now(),
      content: userInput,
      role: 'user',
      imageBytes: sentImage?.bytes,
      imageMime: sentImage?.mime,
    };
    const assistantMsg: Message = {
      id: Date.now() + 1,
      content: '',
      role: 'assistant',
    };

    let convId = currConv?.id;
    if (!convId) {
      const newConv = newConversation(userMsg);
      convId = newConv.id;
      navigateTo(Screen.CHAT, convId);
      addMessageToConversation(convId, assistantMsg);
    } else {
      addMessageToConversation(convId, userMsg);
      addMessageToConversation(convId, assistantMsg);
    }

    if (!loadedModel) {
      throw new Error('loadedModel is null');
    }

    if (sentImage) {
      // Multimodal path: hand-roll the chat format per architecture.
      // We bypass @huggingface/jinja because every VLM's template iterates
      // message.content as a multimodal content list, but our Message only
      // stores strings. Per-model templates also let us guarantee the
      // user-then-assistant boundary so EOS fires cleanly.
      const url = loadedModel.url;
      let formattedPrompt: string;
      if (/LFM2(\.5)?-VL/i.test(url)) {
        // LiquidAI LFM2.5-VL: bos + ChatML-style turns, eos = <|im_end|>
        formattedPrompt =
          `<|startoftext|><|im_start|>user\n` +
          `<__media__>${userInput}<|im_end|>\n` +
          `<|im_start|>assistant\n`;
      } else {
        // SmolVLM (default): "User:" / "Assistant:" with <end_of_utterance>
        formattedPrompt =
          `<|im_start|>User: <__media__>${userInput}<end_of_utterance>\n` +
          `<|im_start|>Assistant:`;
      }
      await createCompletionWithImage(
        formattedPrompt,
        sentImage.bytes,
        (newContent) => {
          editMessageInConversation(convId, assistantMsg.id, newContent);
        }
      );
    } else {
      let formattedChat: string;
      try {
        formattedChat = await formatChat(getWllamaInstance(), [
          ...currHistory,
          userMsg,
        ]);
      } catch (e) {
        alert(
          `Error while formatting chat: ${(e as any)?.message ?? 'unknown'}`
        );
        throw e;
      }
      console.log({ formattedChat });
      await createCompletion(formattedChat, (newContent) => {
        editMessageInConversation(convId, assistantMsg.id, newContent);
      });
    }
    await refreshPerf();
  };

  const onPickImage = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const bytes = new Uint8Array(reader.result as ArrayBuffer);
      if (stagedImage?.previewUrl) URL.revokeObjectURL(stagedImage.previewUrl);
      const previewUrl = URL.createObjectURL(
        new Blob([bytes], { type: file.type || 'image/png' })
      );
      setStagedImage({
        bytes,
        mime: file.type || 'image/png',
        previewUrl,
      });
    };
    reader.readAsArrayBuffer(file);
  };

  const clearStagedImage = () => {
    if (stagedImage?.previewUrl) URL.revokeObjectURL(stagedImage.previewUrl);
    setStagedImage(null);
  };

  return (
    <ScreenWrapper fitScreen>
      <div className="chat-messages grow overflow-auto" id="chat-history">
        <div className="h-10" />

        {currConv ? (
          <>
            {currConv.messages.map((msg) =>
              msg.role === 'user' ? (
                <div className="chat chat-end" key={msg.id}>
                  <div className="chat-bubble">
                    {msg.imageBytes && (
                      <UserImage
                        bytes={msg.imageBytes}
                        mime={msg.imageMime || 'image/png'}
                      />
                    )}
                    {msg.content.length > 0 && (
                      <MarkdownMessage content={msg.content} />
                    )}
                  </div>
                </div>
              ) : (
                <div className="chat chat-start" key={msg.id}>
                  <div className="chat-bubble bg-base-100 text-base-content">
                    {msg.content.length === 0 && isGenerating && (
                      <span className="loading loading-dots"></span>
                    )}
                    {msg.content.length > 0 && (
                      <MarkdownMessage content={msg.content} />
                    )}
                  </div>
                </div>
              )
            )}
          </>
        ) : (
          <div className="pt-24 text-center text-xl">Ask me something 👋</div>
        )}
      </div>
      <div className="flex flex-col input-message py-4">
        {isGenerating && (
          <div className="text-center">
            <button
              className="btn btn-outline btn-sm mb-4"
              onClick={stopCompletion}
            >
              <FontAwesomeIcon icon={faStop} />
              Stop generation
            </button>
          </div>
        )}

        {loadedModel && (
          <>
            {stagedImage && (
              <div className="flex items-center gap-2 mb-2">
                <img
                  src={stagedImage.previewUrl}
                  alt="staged"
                  className="h-16 w-16 object-cover rounded border"
                />
                <button
                  className="btn btn-xs btn-circle"
                  onClick={clearStagedImage}
                  title="Remove image"
                  type="button"
                >
                  <FontAwesomeIcon icon={faXmark} />
                </button>
                <span className="text-xs opacity-70">
                  {(stagedImage.bytes.byteLength / 1024).toFixed(1)} KB
                </span>
              </div>
            )}
            <div className="flex items-end gap-2">
              {visionAvailable && (
                <>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    style={{ display: 'none' }}
                    onChange={onPickImage}
                  />
                  <button
                    type="button"
                    className="btn btn-square btn-ghost"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isGenerating}
                    title="Attach image"
                  >
                    <FontAwesomeIcon icon={faPaperclip} />
                  </button>
                </>
              )}
              <textarea
                className="textarea textarea-bordered w-full"
                placeholder={
                  visionAvailable
                    ? 'Your message... (paperclip to attach an image)'
                    : 'Your message...'
                }
                disabled={isGenerating}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.keyCode == 13 && e.shiftKey == false) {
                    e.preventDefault();
                    onSubmit();
                  }
                }}
              />
            </div>
            <div className="mt-3 text-xs">
              <div className="flex items-center justify-between">
                <div>
                  {perfError && (
                    <div className="text-error">Error: {perfError}</div>
                  )}
                  {!perfError && (
                    <div>
                      Prefill:{' '}
                      {formatTokPerSec(perfData.n_p_eval, perfData.t_p_eval_ms)}{' '}
                      tok/s, Decode:{' '}
                      {formatTokPerSec(perfData.n_eval, perfData.t_eval_ms)}{' '}
                      tok/s
                    </div>
                  )}
                </div>
                <button
                  className="btn btn-xs btn-outline"
                  disabled={perfBusy || isGenerating}
                  onClick={resetPerf}
                >
                  Reset
                </button>
              </div>
            </div>
          </>
        )}

        {!loadedModel && <WarnNoModel />}

        <small className="text-center mx-auto opacity-70 pt-2">
          wllama may generate inaccurate information. Use with your own risk.
        </small>
      </div>
    </ScreenWrapper>
  );
}

function WarnNoModel() {
  const { navigateTo } = useWllama();

  return (
    <div role="alert" className="alert">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        className="h-6 w-6 shrink-0 stroke-current"
        fill="none"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
      <span>Model is not loaded</span>
      <div>
        <button
          className="btn btn-sm btn-primary"
          onClick={() => navigateTo(Screen.MODEL)}
        >
          Select model
        </button>
      </div>
    </div>
  );
}

function UserImage({ bytes, mime }: { bytes: Uint8Array; mime: string }) {
  const url = useMemo(() => {
    return URL.createObjectURL(new Blob([bytes as BlobPart], { type: mime }));
  }, [bytes, mime]);
  useEffect(() => {
    return () => URL.revokeObjectURL(url);
  }, [url]);
  return (
    <img
      src={url}
      alt="attachment"
      className="max-h-48 max-w-xs rounded mb-2 block"
    />
  );
}

const chatScrollToBottom = () => {
  const elem = document.getElementById('chat-history');
  elem?.scrollTo({
    top: elem.scrollHeight,
    behavior: 'smooth',
  });
};
