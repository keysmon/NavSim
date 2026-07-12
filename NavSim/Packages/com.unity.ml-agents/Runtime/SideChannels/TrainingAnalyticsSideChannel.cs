using System;
using Unity.MLAgents.Analytics;
// NavSim: this side channel decodes Protobuf training-analytics messages, which IL2CPP cannot
// convert for WebGL. The channel only ever receives messages from a connected trainer (never in
// an inference-only player), so the decode body is gated out on non-training platforms.
#if UNITY_EDITOR || UNITY_STANDALONE
using Unity.MLAgents.CommunicatorObjects;
#endif

namespace Unity.MLAgents.SideChannels
{
    /// <summary>
    /// Side Channel implementation for recording which training features are being used.
    /// </summary>
    internal class TrainingAnalyticsSideChannel : SideChannel
    {
        const string k_TrainingAnalyticsConfigId = "b664a4a9-d86f-5a5f-95cb-e8353a7e8356";

        /// <summary>
        /// Initializes the side channel. The constructor is internal because only one instance is
        /// supported at a time, and is created by the Academy.
        /// </summary>
        internal TrainingAnalyticsSideChannel()
        {
            ChannelId = new Guid(k_TrainingAnalyticsConfigId);
        }

        /// <inheritdoc/>
        protected override void OnMessageReceived(IncomingMessage msg)
        {
#if UNITY_EDITOR || UNITY_STANDALONE
            Google.Protobuf.WellKnownTypes.Any anyMessage = null;
            try
            {
                anyMessage = Google.Protobuf.WellKnownTypes.Any.Parser.ParseFrom(msg.GetRawBytes());
            }
            catch (Google.Protobuf.InvalidProtocolBufferException)
            {
                // Bad message, nothing we can do about it, so just ignore.
                return;
            }

            if (anyMessage.Is(TrainingEnvironmentInitialized.Descriptor))
            {
                var envInitProto = anyMessage.Unpack<TrainingEnvironmentInitialized>();
                var envInitEvent = envInitProto.ToTrainingEnvironmentInitializedEvent();
                TrainingAnalytics.TrainingEnvironmentInitialized(envInitEvent);
            }
            else if (anyMessage.Is(TrainingBehaviorInitialized.Descriptor))
            {
                var behaviorInitProto = anyMessage.Unpack<TrainingBehaviorInitialized>();
                var behaviorTrainingEvent = behaviorInitProto.ToTrainingBehaviorInitializedEvent();
                TrainingAnalytics.TrainingBehaviorInitialized(behaviorTrainingEvent);
            }
            // Don't do anything for unknown types, since the user probably can't do anything about it.
#endif
        }
    }
}
