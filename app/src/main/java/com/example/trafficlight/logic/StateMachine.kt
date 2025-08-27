package com.example.trafficlight.logic

import com.example.trafficlight.inference.ClassificationResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class TrafficLightState {
    RED,
    YELLOW, 
    GREEN,
    OFF,
    UNKNOWN
}

data class TrafficLightEvent(
    val state: TrafficLightState,
    val confidence: Float,
    val timestamp: Long = System.currentTimeMillis()
)

class StateMachine {
    private val _currentState = MutableStateFlow(TrafficLightState.UNKNOWN)
    val currentState: StateFlow<TrafficLightState> = _currentState.asStateFlow()
    
    private val _shouldAnnounce = MutableStateFlow(false)
    val shouldAnnounce: StateFlow<Boolean> = _shouldAnnounce.asStateFlow()
    
    private val votingWindow = mutableListOf<TrafficLightEvent>()
    private val maxWindowSize = 5
    private val confidenceThreshold = 0.8f
    private val stateChangeThreshold = 0.6f
    
    private var lastAnnouncedState = TrafficLightState.UNKNOWN
    private var lastStateChangeTime = 0L
    private val minStateChangeCooldown = 2000L
    
    fun processClassification(result: ClassificationResult) {
        val state = when (result.classId) {
            ClassificationResult.RED -> TrafficLightState.RED
            ClassificationResult.YELLOW -> TrafficLightState.YELLOW
            ClassificationResult.GREEN -> TrafficLightState.GREEN
            ClassificationResult.OFF -> TrafficLightState.OFF
            else -> TrafficLightState.UNKNOWN
        }
        
        val event = TrafficLightEvent(state, result.confidence)
        addVote(event)
        
        val consensusState = determineConsensusState()
        updateState(consensusState)
    }
    
    private fun addVote(event: TrafficLightEvent) {
        votingWindow.add(event)
        
        if (votingWindow.size > maxWindowSize) {
            votingWindow.removeAt(0)
        }
        
        cleanOldVotes()
    }
    
    private fun cleanOldVotes() {
        val currentTime = System.currentTimeMillis()
        val maxAge = 1500L
        
        votingWindow.removeAll { currentTime - it.timestamp > maxAge }
    }
    
    private fun determineConsensusState(): TrafficLightState {
        if (votingWindow.isEmpty()) return TrafficLightState.UNKNOWN
        
        val stateVotes = mutableMapOf<TrafficLightState, Float>()
        var totalWeight = 0f
        
        for (vote in votingWindow) {
            val weight = vote.confidence
            stateVotes[vote.state] = (stateVotes[vote.state] ?: 0f) + weight
            totalWeight += weight
        }
        
        if (totalWeight == 0f) return TrafficLightState.UNKNOWN
        
        val bestState = stateVotes.maxByOrNull { it.value }?.key ?: TrafficLightState.UNKNOWN
        val bestStateRatio = (stateVotes[bestState] ?: 0f) / totalWeight
        
        return if (bestStateRatio >= stateChangeThreshold) {
            bestState
        } else {
            _currentState.value
        }
    }
    
    private fun updateState(newState: TrafficLightState) {
        val currentTime = System.currentTimeMillis()
        val currentStateValue = _currentState.value
        
        if (newState != currentStateValue && newState != TrafficLightState.UNKNOWN) {
            if (currentTime - lastStateChangeTime >= minStateChangeCooldown) {
                _currentState.value = newState
                lastStateChangeTime = currentTime
                
                checkShouldAnnounce(newState)
            }
        }
    }
    
    private fun checkShouldAnnounce(newState: TrafficLightState) {
        val shouldAnnounceNow = when {
            newState == lastAnnouncedState -> false
            newState == TrafficLightState.UNKNOWN || newState == TrafficLightState.OFF -> false
            newState == TrafficLightState.RED -> true
            newState == TrafficLightState.GREEN -> true
            newState == TrafficLightState.YELLOW -> {
                val highConfidenceVotes = votingWindow.filter { 
                    it.state == TrafficLightState.YELLOW && it.confidence >= confidenceThreshold 
                }
                highConfidenceVotes.size >= 2
            }
            else -> false
        }
        
        if (shouldAnnounceNow) {
            lastAnnouncedState = newState
            _shouldAnnounce.value = true
        }
    }
    
    fun getStateConfidence(): Float {
        if (votingWindow.isEmpty()) return 0f
        
        val currentStateValue = _currentState.value
        val relevantVotes = votingWindow.filter { it.state == currentStateValue }
        
        return if (relevantVotes.isNotEmpty()) {
            relevantVotes.map { it.confidence }.average().toFloat()
        } else {
            0f
        }
    }
    
    fun getVotingWindowInfo(): String {
        if (votingWindow.isEmpty()) return "No votes"
        
        val stateCounts = votingWindow.groupingBy { it.state }.eachCount()
        return stateCounts.entries.joinToString(", ") { "${it.key}:${it.value}" }
    }
    
    fun acknowledgeAnnouncement() {
        _shouldAnnounce.value = false
    }
    
    fun reset() {
        votingWindow.clear()
        _currentState.value = TrafficLightState.UNKNOWN
        _shouldAnnounce.value = false
        lastAnnouncedState = TrafficLightState.UNKNOWN
        lastStateChangeTime = 0L
    }
    
    fun getCurrentStateString(): String {
        return when (_currentState.value) {
            TrafficLightState.RED -> "紅燈"
            TrafficLightState.YELLOW -> "黃燈"
            TrafficLightState.GREEN -> "綠燈"
            TrafficLightState.OFF -> "燈號關閉"
            TrafficLightState.UNKNOWN -> "未知"
        }
    }
}