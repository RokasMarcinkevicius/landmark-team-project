import React, { useState, useEffect } from 'react';
import {
    View,
    StyleSheet,
    FlatList,
    RefreshControl,
    Text,
} from 'react-native';
import { Card, ActivityIndicator } from 'react-native-paper';
import { getHistory } from '../services/api';
import { HistoryRecord, LANDMARK_NAMES } from '../types';

export default function HistoryScreen() {
    const [history, setHistory] = useState<HistoryRecord[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [refreshing, setRefreshing] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const loadHistory = async () => {
        try {
            setError(null);
            const data = await getHistory(50);
            setHistory(data);
        } catch (err: any) {
            console.error('Failed to load history:', err);
            setError('Failed to load history. Make sure the backend is running.');
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    useEffect(() => {
        loadHistory();
    }, []);

    const onRefresh = () => {
        setRefreshing(true);
        loadHistory();
    };

    const getDisplayName = (label: string) => {
        return LANDMARK_NAMES[label] || label;
    };

    const formatDate = (timestamp: string) => {
        const date = new Date(timestamp);
        return date.toLocaleString();
    };

    const formatConfidence = (confidence: number) => {
        return `${(confidence * 100).toFixed(1)}%`;
    };

    const renderItem = ({ item }: { item: HistoryRecord }) => (
        <Card style={styles.card}>
            <Card.Content>
                <View style={styles.cardHeader}>
                    <Text style={styles.landmark}>
                        {getDisplayName(item.topPrediction)}
                    </Text>
                    <Text style={styles.confidence}>
                        {formatConfidence(item.topConfidence)}
                    </Text>
                </View>

                <Text style={styles.date}>{formatDate(item.timestamp)}</Text>

                <View style={styles.infoRow}>
                    <Text style={styles.infoText}>
                        ⚡ {item.inferenceTimeMs}ms
                    </Text>
                    <Text style={styles.infoText}>
                        📁 {item.imageSizeMb.toFixed(2)} MB
                    </Text>
                </View>
            </Card.Content>
        </Card>
    );

    if (loading) {
        return (
            <View style={styles.centerContainer}>
                <ActivityIndicator size="large" color="#6200ea" />
                <Text style={styles.loadingText}>Loading history...</Text>
            </View>
        );
    }

    if (error) {
        return (
            <View style={styles.centerContainer}>
                <Text style={styles.errorText}>❌ {error}</Text>
            </View>
        );
    }

    if (history.length === 0) {
        return (
            <View style={styles.centerContainer}>
                <Text style={styles.emptyText}>No predictions yet.</Text>
                <Text style={styles.emptySubtext}>
                    Take your first photo to get started!
                </Text>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <FlatList
                data={history}
                renderItem={renderItem}
                keyExtractor={(item) => item.id.toString()}
                contentContainerStyle={styles.listContent}
                refreshControl={
                    <RefreshControl
                        refreshing={refreshing}
                        onRefresh={onRefresh}
                        colors={['#6200ea']}
                        tintColor="#6200ea"
                    />
                }
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
    },
    centerContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    listContent: {
        padding: 15,
    },
    card: {
        marginBottom: 15,
        elevation: 2,
    },
    cardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
    },
    landmark: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#333',
        flex: 1,
    },
    confidence: {
        fontSize: 16,
        fontWeight: '600',
        color: '#6200ea',
    },
    date: {
        fontSize: 14,
        color: '#666',
        marginBottom: 10,
    },
    infoRow: {
        flexDirection: 'row',
        gap: 20,
    },
    infoText: {
        fontSize: 12,
        color: '#999',
    },
    loadingText: {
        marginTop: 10,
        fontSize: 16,
        color: '#666',
    },
    errorText: {
        fontSize: 16,
        color: '#d32f2f',
        textAlign: 'center',
    },
    emptyText: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#666',
        marginBottom: 10,
    },
    emptySubtext: {
        fontSize: 14,
        color: '#999',
        textAlign: 'center',
    },
});